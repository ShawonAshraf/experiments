import os
from nltk.tokenize import word_tokenize as tokenize
from nltk.corpus import stopwords
from string import punctuation
from tqdm import tqdm


corpus_root = os.path.join(os.getcwd(), "review_polarity", "txt_sentoken")
catgeories = ["pos", "neg"]

# stopwords for english
ignore = stopwords.words("english")

"""
One file contains one text instance in the corpus
"""


# just read all files for a category
def read_text_files(path):
    file_list = os.listdir(path)
    texts = []

    for fname in file_list:
        fpath = os.path.join(path, fname)

        f = open(fpath, mode="r")
        lines = f.read()
        texts.append(lines)
        f.close()

    return texts


def remove_stopwords(tokens):
    x = [token for token in tokens if token not in ignore]
    return x


def remove_punctuation(tokens):
    x = [token for token in tokens if token not in punctuation]
    return x


def clean(tokens, remove_sw=True):
    if remove_sw:
        x = remove_stopwords(tokens)
    x = remove_punctuation(x)

    return x


# label -> 1 for pos and 0 for neg
def prepare_corpus(remove_sw=True):
    # dataset is a list of tuples
    # (label, tokens)
    corpus = list()

    # idx -> label
    categories = ["neg", "pos"]
    for idx, category in enumerate(categories):
        # root + category path
        path = os.path.join(corpus_root, category)

        texts = read_text_files(path)

        for i in tqdm(range(len(texts)), desc="prepare_corpus"):
            text = texts[i]
            # tokenize
            tokens = tokenize(text)

            # clean
            tokens = clean(tokens=tokens, remove_sw=remove_sw)

            # append
            corpus.append((idx, tokens))

    return corpus
