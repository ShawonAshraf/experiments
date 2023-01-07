# %%

from sklearn.metrics import classification_report
from pytorch_lightning.strategies import DeepSpeedStrategy
from torch.utils.data import DataLoader
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
import deepspeed
from torch.optim import Adam
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoModel
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import LongformerTokenizer, LongformerModel, LongformerForSequenceClassification
import torch
import numpy as np
import os
from tqdm import tqdm
from corpus import download_and_unzip, catgeories, read_text_files, corpus_root

download_and_unzip()

# %%

reviews = []
labels = []

# we can't use the previous tokenizers here
# idx 0 -> neg, 1 -> pos
for idx, cat in enumerate(catgeories):
    path = os.path.join(corpus_root, cat)
    texts = read_text_files(path)

    for i in tqdm(range(len(texts)), desc="prepare_corpus"):
        text = texts[i]
        reviews.append(text)
        labels.append(idx)

# %%

# %%
pretrained_model_name = "allenai/longformer-base-4096"

tokenizer = LongformerTokenizer.from_pretrained(pretrained_model_name)

# %%

x_train, x_test, y_train, y_test = train_test_split(
    reviews, labels, random_state=42, train_size=0.8
)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, train_size=0.8, random_state=42)

# %%

# custom dataset


class PolarityReviewDataset(Dataset):

    def __init__(self, reviews, labels, tokenizer):
        self.reviews = reviews
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        label = self.labels[idx]
        label = torch.tensor(label)
        label = torch.nn.functional.one_hot(label, num_classes=2)

        # encode review text
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=2048,
            truncation=True,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        )

        return {
            "text": review,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding['attention_mask'].flatten(),
            "label": label.float()
        }


# %%
training_dataset = PolarityReviewDataset(x_train, y_train, tokenizer)
val_dataset = PolarityReviewDataset(x_val, y_val, tokenizer)

# %%


class SentiBERT(pl.LightningModule):
    def __init__(self, model_path=pretrained_model_name):
        super(SentiBERT, self).__init__()

        self.model_path = model_path

        self.longformer = LongformerModel.from_pretrained(model_path)
        # self.longformer = LongformerForSequenceClassification.from_pretrained(model_path, num_labels=2)
        self.linear = nn.Linear(768, 2, dtype=torch.float16)
        self.softmax = nn.Softmax(dim=-1)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask):
        out = self.longformer(input_ids, attention_mask)

        out = out.pooler_output

        out = self.linear(out)

        out = self.softmax(out)

        return out

    # def configure_sharded_model(self):
    #     self.longformer = LongformerModel.from_pretrained(
    #         self.model_path, ignore_mismatched_sizes=True)

    def configure_optimizers(self):
        # return Adam(self.parameters(), lr=2e-5)
        return DeepSpeedCPUAdam(self.parameters(), lr=2e-5)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attn_masks = batch["attention_mask"]
        labels = batch["label"]

        logits = self(input_ids, attn_masks)
        loss = self.criterion(logits, labels)

        logs = {"train_loss": loss}
        return {
            "loss": loss,
            "log": logs
        }

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attn_masks = batch["attention_mask"]
        labels = batch["label"]

        logits = self(input_ids, attn_masks)
        loss = self.criterion(logits, labels)

        self.log("val_loss", loss, prog_bar=True)


senti_bert = SentiBERT()
# # test with sample input
# sample_inp = tokenizer.encode_plus(
#     "This is a sample text", return_tensors="pt")
# logits = senti_bert(**sample_inp)

# logits

# %%

batch_size = 10
# torch.backends.cudnn.benchmark = True

# loader from custom dataset
train_loader = DataLoader(
    training_dataset, shuffle=True, batch_size=batch_size,
    num_workers=4)
val_loader = DataLoader(val_dataset, shuffle=False,
                        batch_size=batch_size, num_workers=4)

# deepspeed strategy
strategy = DeepSpeedStrategy(
    stage=3,
    offload_optimizer=True,
    offload_parameters=True,
    offload_optimizer_device="cpu",
    offload_params_device="cpu",
    cpu_checkpointing=True,
    pin_memory=True
)

trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=2,
    strategy=strategy,
    precision=16,
    log_every_n_steps=50)

trainer.fit(senti_bert, train_loader, val_loader)

# %%
# test data is a list of reviews as strings


# def classify_sentiment(model, test_data, tokenizer):
#     prediction = []
#     # switch model mode
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = model.to(device)
#     model.eval()
#     with torch.no_grad():

#         for i in tqdm(range(len(test_data)), desc="inference"):
#             review = test_data[i]

#             # encode data
#             encoded = tokenizer.encode_plus(
#                 review,
#                 add_special_tokens=True,
#                 max_length=1600,
#                 truncation=True,
#                 return_token_type_ids=False,
#                 padding="max_length",
#                 return_attention_mask=True,
#                 return_tensors="pt"
#             )

#             # unpack
#             input_ids = encoded["input_ids"].to(device)
#             attention_mask = encoded["attention_mask"].to(device)

#             # forward pass
#             out = model(input_ids, attention_mask)
#             _, idx = torch.max(out, dim=-1)
#             # dear pytorch team, find a easier wrapper please!
#             pred = idx.detach().cpu().numpy()

#             # add to list
#             prediction.append(pred)

#     return np.array(prediction)

# # %%
# # x = torch.Tensor([[0.5367, 0.4633]])
# # v, idx = torch.max(x, dim=-1)

# # idx.numpy()


# # %%
# y_pred = classify_sentiment(senti_bert, x_test, tokenizer)

# # %%
# y_pred[:10]

# # %%
# # y_pred = y_pred.reshape(-1, 1)
# # y_pred.shape

# # %%
# y_test = np.array(y_test).reshape(-1, 1)
# y_test[:10]

# # %%

# print(classification_report(y_pred=y_pred, y_true=y_test))

# # %%
