import torch
from transformers import ReformerTokenizer
from torch.utils.data import Dataset


class PolarityReviewDataset(Dataset):

    def __init__(self, reviews, labels, max_length=1490, model_name="google/reformer-crime-and-punishment"):
        self.MAX_LEN = max_length
        self.tokenizer = ReformerTokenizer.from_pretrained(model_name)

        self.reviews = reviews
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        label = self.labels[idx]

        # encode review text
        encoding = self.tokenizer(
            review, return_tensors="pt")

        return {
            "text": review,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding['attention_mask'].flatten(),
            "label": torch.tensor(label)
        }
