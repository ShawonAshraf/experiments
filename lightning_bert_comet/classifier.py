import torch
from torch.nn import functional as F
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule

from transformers import AutoModel

from torch.optim import Adam


class SentiBERT(LightningModule):
    def __init__(self, model_name):
        super(SentiBERT, self).__init__()

        self.bert = AutoModel.from_pretrained(model_name)
        # 768 for BERT, 1 for binary classification
        self.linear = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

        self.loss_fn = nn.BCELoss()

    def forward(self, x):
        input_ids, attention_mask = x
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        out = out.pooler_output

        out = self.linear(out)
        out = self.sigmoid(out)

        return out

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=2e-5)

    def training_step(self, batch, batch_idx):
        td = batch

        input_ids = td["input_ids"]
        attention_mask = td["attention_mask"]
        label = td["label"]

        out = self((input_ids, attention_mask))
        logits, _ = torch.max(out, dim=1)
        loss = self.loss_fn(logits, label.float())

        logs = {"train_loss": loss}

        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        td = batch

        input_ids = td["input_ids"]
        attention_mask = td["attention_mask"]
        label = td["label"]

        out = self((input_ids, attention_mask))
        logits, _ = torch.max(out, dim=1)
        loss = self.loss_fn(logits, label.float())

        self.log("val_loss", loss, prog_bar=True)
