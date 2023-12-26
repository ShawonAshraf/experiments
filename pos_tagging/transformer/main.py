from datasets import load_dataset
import time

import torch
from torch.utils.data import DataLoader
import lightning.pytorch as L

from tagdataset import TagDataset
from model import TagFormer
from utils import create_label_to_idx, create_train_validation_splits, create_word_indices, encode_data_instance


torch.manual_seed(2023)
torch.set_float32_matmul_precision('high')



if __name__ == "__main__":
    # ======================
    dataset_name = "batterydata/pos_tagging"
    training_dataset = load_dataset(dataset_name, split="train")
    test_dataset = load_dataset(dataset_name, split="test")
    
    word_to_idx = create_word_indices(training_dataset)
    label_to_idx = create_label_to_idx(training_dataset)
        
    # ======================
    trainset = map(lambda data: encode_data_instance(
            data, word_to_idx, label_to_idx), training_dataset)
    trainset = list(trainset)
    testset = map(lambda data: encode_data_instance(
        data, word_to_idx, label_to_idx), test_dataset)
    testset = list(testset)
    
    # ======================
    trainset_indices, validation_indices = create_train_validation_splits(
        trainset, 0.3)
    assert len(trainset_indices) + len(validation_indices) == len(trainset)
    
    # ======================
    train_loader = DataLoader(
        TagDataset(trainset_indices, trainset), batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(
        TagDataset(validation_indices, trainset), batch_size=128, shuffle=False, num_workers=4)
    test_loader = DataLoader(
        TagDataset(None, testset), batch_size=128, shuffle=False, num_workers=4)
    
    # ======================
    model = TagFormer(vocab_size=len(word_to_idx), embedding_dim=512, n_labels=300, 
                      pad_idx=1, nhead=8, n_encoder_layers=6)
    
    # with torch.no_grad():
    #     for batch in train_loader:
    #         words, labels = batch
    #         logits = model(words)
    #         print(logits.shape)

    #         break


    
    trainer = L.Trainer(
                        max_epochs=100,
                        accelerator="gpu",
                        devices=1,
                        precision="bf16-mixed",
                        log_every_n_steps=50)
    
    trainer.fit(model, train_loader, val_loader)
    

    
    
