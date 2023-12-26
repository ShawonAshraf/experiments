import torch
from torch.utils.data import Dataset
import numpy as np

torch.manual_seed(2023)


class TagDataset(Dataset):
    def __init__(self, indices, dataset) -> None:
        self.indices = indices
        self.dataset = dataset

    def __len__(self):
        if self.indices is None:
            # this is for the test case
            return len(self.dataset)
        else:
            return len(self.indices)

    def __getitem__(self, index):
        if self.indices is None:
            idx = index
        else:
            idx = self.indices[index]

        data = self.dataset[idx]

        # padding to 300
        # pad token idx is 1
        words = np.ones((300, ), dtype=np.int32)
        words[:len(data["words"])] = data["words"]

        labels = np.ones((300, ), dtype=np.int32)
        labels[:len(data["labels"])] = data["labels"]

        return torch.from_numpy(words).long(), torch.from_numpy(labels).long()
