import torch
import torch.nn as nn
from helpers import prepare_data
from model import SentiBERT
from trainer import train

if __name__ == "__main__":
    train_loader, val_loader, test_dataset = prepare_data(
        split_ratio=0.8, num_workers=32, batch_size=32)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    model = SentiBERT()
    model = model.to(device=device)

    lr = 2e-5
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train(model, train_loader, val_loader,
          epochs=100, optimizer=optimizer, loss_fn=loss_fn, device=device)
