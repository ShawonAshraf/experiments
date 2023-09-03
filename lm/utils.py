import os
import urllib
import sys

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

os.makedirs("data", exist_ok=True)

def download(url=url):
    file_name = url.split("/")[-1]
    download_path = os.path.join("./data", file_name)
    
    if os.path.exists(download_path):
        print("Already downloaded!")
        
    else:
        # ============================================ download
        print("Downloading, sit tight!")

        def _progress(count, block_size, total_size):
            sys.stdout.write(
                f"\r>> Downloading {file_name} {float(count * block_size) / float(total_size) * 100.0}%")
            sys.stdout.flush()

        file_path, _ = urllib.request.urlretrieve(
            url, download_path, _progress)
        print()
        print(
            f"Successfully downloaded {file_name} {os.stat(file_path).st_size} bytes")


def read_data(file_path="./data/input.txt"):
    assert os.path.exists(file_path)
    
    with open(file_path, "r") as f:
        data = f.readlines()
        
    # remove new line escape sequences
    data = [d for d in data if d != "\n"]
    data = [d.replace("\n", "") for d in data]

    return data

def tokenize_sentence(sentence):
    s = sentence.split(" ")
    return s

def batch_tokenize_sentences(sentences):
    return [tokenize_sentence(s) for s in sentences]
