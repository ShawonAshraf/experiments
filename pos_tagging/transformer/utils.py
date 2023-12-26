import numpy as np


def create_word_indices(dataset):
    unique_words = set()
    word_to_idx = dict()
    # add an out of vocab token
    oov_token = "<OOV>"
    pad_token = "<PAD>"
    word_to_idx[oov_token] = 0
    word_to_idx[pad_token] = 1

    # find unique words
    for data in dataset:
        words = data["words"]
        for w in words:
            unique_words.add(w)

    # add index to them
    for idx, uw in enumerate(list(unique_words)):
        word_to_idx[uw] = idx + 2  # since oov is at 0 and pad at 1

    return word_to_idx


def create_label_to_idx(dataset):
    unique_labels = set()
    label_to_idx = dict()
    # add an out of vocab token
    oov_token = "<OOV>"
    pad_token = "<PAD>"
    label_to_idx[oov_token] = 0
    label_to_idx[pad_token] = 1

    # find the labels
    for data in dataset:
        labels = data["labels"]
        for l in labels:
            unique_labels.add(l)

    # index
    for idx, label in enumerate(list(unique_labels)):
        label_to_idx[label] = idx + 2

    return label_to_idx


def encode_data_instance(data, word_to_idx, label_to_idx):
    words = [
        word_to_idx.get(word, word_to_idx["<OOV>"]) for word in data["words"]
    ]

    labels = [
        label_to_idx[label] for label in data["labels"]
    ]

    return {
        "words": words,
        "labels": labels
    }


def create_train_validation_splits(trainset, validation_ratio):
    validation_set_size = int(len(trainset) * validation_ratio)
    validation_indices = np.random.choice(
        len(trainset), replace=False, size=validation_set_size).tolist()

    # now to separate trainset indices
    trainset_indices = [i for i in range(
        len(trainset)) if i not in validation_indices]

    return trainset_indices, validation_indices
