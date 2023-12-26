from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
import lightning.pytorch as L
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import torch.optim as optim
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)



class TagFormer(L.LightningModule):
    def __init__(self, vocab_size, embedding_dim, n_labels, pad_idx, nhead, n_encoder_layers) -> None:
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.pad_idx = pad_idx
        self.n_labels = n_labels
        self.nhead = nhead
        self.n_encoder_layers = n_encoder_layers
        self.save_hyperparameters()
        
        # modules
        self.embedding = nn.Embedding(vocab_size, embedding_dim, pad_idx)
        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embedding_dim, nhead=self.nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_encoder_layers)
        
        # fc
        self.fc = nn.Linear(512, self.n_labels)
        self.dropout = nn.Dropout(0.1)
        
        # self.__custom_init()
        self.embedding.weight.data[self.pad_idx] = torch.zeros(self.embedding_dim, )
        
        self.pe = PositionalEncoding(512)
        
        
    def __custom_init(self):
        for p in self.parameters():
            nn.init.normal_(p.data, mean=0, std=0.1)
            
    
        
    def forward(self, x) -> Any:
        out = self.embedding(x) * math.sqrt(512)
        # print("embed: ", out.size())
        
        out = rearrange(out, "batch seq embed -> seq batch embed")
        out = self.pe(out)
        # expected shape : seq batch embed
        # print("pe: ", out.size())
        
        out = self.encoder(out)
        # expected shape : seq batch embed
        # print("enc: ", out.size())
        
        out = rearrange(out, "seq batch embed -> batch seq embed")
        out = self.fc(out)
        # dropout before relu?
        # https://sebastianraschka.com/faq/docs/dropout-activation.html
        out = self.dropout(out)
        out = F.leaky_relu(out)
        # print("fc: ", out.size())
        
        
        return out
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        return optim.AdamW(self.parameters())
    
    def compute_loss(self, batch):
        words, labels = batch
        logits = self(words)
        
        loss = F.cross_entropy(logits, labels, ignore_index=self.pad_idx)
        return loss
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss = self.compute_loss(batch)

        self.log("Loss/Train", loss, prog_bar=True)

        return {
            "loss": loss,
            "log": {
                "Loss/Train": loss
            }
        }

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        loss = self.compute_loss(batch)

        self.log("Loss/Validation", loss, prog_bar=True)

        return {
            "val_loss": loss,
            "log": {
                "Loss/Validation": loss
            }
        }

        
    
    
if __name__ == "__main__":
    model = TagFormer(50000, 512, 18, 1, 8, 6)
    
    sample_input = torch.stack([torch.arange(300), torch.arange(300)], dim=0)
    with torch.no_grad():
        out = model(sample_input)
        
        
        
