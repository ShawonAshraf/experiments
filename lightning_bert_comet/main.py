from pytorch_lightning.loggers import CometLogger
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from classifier import SentiBERT
from datasets import PolarityReviewDataset

from transformers import AutoTokenizer

from sklearn.model_selection import train_test_split

from corpus import *
from tqdm import tqdm
import os

torch.backends.cudnn.benchmark = True

batch_size = 32
model_name = "bert-base-cased"
MAX_LEN = 512

download_and_unzip()

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

tokenizer = AutoTokenizer.from_pretrained(model_name)
x_train, x_test, y_train, y_test = train_test_split(
    reviews, labels, random_state=0, train_size=0.8
)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, train_size=0.8, random_state=0)


training_dataset = PolarityReviewDataset(x_train, y_train, tokenizer, MAX_LEN)
val_dataset = PolarityReviewDataset(x_val, y_val, tokenizer, MAX_LEN)


train_loader = DataLoader(
    training_dataset, shuffle=True, batch_size=batch_size, num_workers=32, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size,
                        num_workers=32, pin_memory=True)

model = SentiBERT(model_name)
trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=2,
    precision=16,
    log_every_n_steps=10)

trainer.fit(model, train_loader, val_loader)
