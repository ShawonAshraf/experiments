import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset


class PolarityReviewDataset(Dataset):

    def __init__(self, reviews, labels, max_length=256, model_name="bert-base-cased"):
        self.MAX_LEN = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.reviews = reviews
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        review = self.reviews[idx]
        label = self.labels[idx]

        # encode review text
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.MAX_LEN,
            truncation=True,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors="pt"
        )

        return {
            "text": review,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding['attention_mask'].flatten(),
            "label": torch.tensor(label)
        }
