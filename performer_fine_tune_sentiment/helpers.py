import os

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from corpus import download_and_unzip, catgeories, read_text_files, corpus_root
from dataset import PolarityReviewDataset


def prepare_data(split_ratio, num_workers, batch_size):
    # download dataset
    download_and_unzip()

    reviews = list()
    labels = list()

    # idx 0 -> neg, 1 -> pos
    for idx, cat in enumerate(catgeories):
        path = os.path.join(corpus_root, cat)
        texts = read_text_files(path)

        for i in tqdm(range(len(texts)), desc="prepare_data"):
            text = texts[i]
            reviews.append(text)
            labels.append(idx)

    # train test validation split
    x_train, x_test, y_train, y_test = train_test_split(
        reviews, labels, random_state=42, train_size=split_ratio
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, train_size=split_ratio, random_state=42)

    # datasets
    training_dataset = PolarityReviewDataset(x_train, y_train)
    val_dataset = PolarityReviewDataset(x_val, y_val)
    test_dataset = PolarityReviewDataset(x_test, y_test)

    # dataloaders
    train_loader = DataLoader(training_dataset, shuffle=True,
                              batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, shuffle=False,
                            batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader, test_dataset
