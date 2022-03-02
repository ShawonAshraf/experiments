import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained(
    "google/reformer-crime-and-punishment")
model = AutoModel.from_pretrained("google/reformer-crime-and-punishment")

text = "GPUs are expensive these days!"
ins = tokenizer(text, return_tensors="pt")
# print(ins)

out = model(**ins)
print(out.last_hidden_state.shape)
