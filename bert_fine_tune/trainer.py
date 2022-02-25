import torch
import numpy as np
from tqdm import tqdm


def train(model, train_loader, val_loader, epochs, optimizer, loss_fn, device):
    print_counter = 0  # validation for each 10th count

    for e in range(epochs):
        model.train()
        for td in tqdm(train_loader):
            print_counter += 1

            # unpack data and send to device
            input_ids = td["input_ids"]
            input_ids = input_ids.to(device)

            attention_mask = td["attention_mask"].to(device)
            attention_mask = attention_mask.to(device)

            label = td["label"]
            label = label.to(device)

            # zero gradients
            model.zero_grad()

            # forward pass
            output = model(input_ids, attention_mask)

            # the max probability based class
            output, _ = torch.max(output, dim=1)

            # backprop
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()

            # log loss
            if print_counter % 10 == 0:
                validation_losses = []

                model.eval()  # switch mode
                with torch.no_grad():
                    for td_val in val_loader:
                        # unpack data and send to device
                        input_ids = td_val["input_ids"]
                        input_ids = input_ids.to(device)

                        attention_mask = td_val["attention_mask"]
                        attention_mask = attention_mask.to(device)

                        label = td_val["label"]
                        label = label.to(device)

                        # repeat same steps from forward pass
                        out = model(input_ids, attention_mask)
                        out, _ = torch.max(out, dim=1)
                        val_loss = loss_fn(out, label)

                        # add loss to validation losses
                        validation_losses.append(val_loss.item())
                    print(
                        f"\nEpoch: {e + 1}/{epochs}\tStep: {print_counter}\tTrain Loss: {loss.item()}\tValidation Loss: {np.mean(validation_losses)}")
                # switch back mode
                model.train()
