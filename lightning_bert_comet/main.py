from comet_ml import Experiment
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

batch_size = 16
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
    reviews, labels, random_state=42, train_size=0.8
)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, train_size=0.8, random_state=42)


training_dataset = PolarityReviewDataset(x_train, y_train, tokenizer, MAX_LEN)
val_dataset = PolarityReviewDataset(x_val, y_val, tokenizer, MAX_LEN)


train_loader = DataLoader(
    training_dataset, shuffle=True, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

comet_api_key = os.getenv("COMET_API_KEY")

comet_logger = CometLogger(
    api_key=comet_api_key,
    project_name='lightning-bert-comet',
    experiment_name='exp-4-cpu',
    auto_output_logging="simple",
)

model = SentiBERT(model_name)
trainer = pl.Trainer(max_epochs=2, logger=comet_logger)
trainer.fit(model, train_loader, val_loader)
