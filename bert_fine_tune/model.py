import torch
import torch.nn as nn
from transformers import AutoModel


class SentiBERT(nn.Module):
    def __init__(self, model_name="bert-base-cased"):
        super(SentiBERT, self).__init__()

        self.bert = AutoModel.from_pretrained(model_name)

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # get the hidden state of the token CLS
        out = out.pooler_output
        y = self.classifier(out)

        return y
