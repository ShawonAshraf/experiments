import torch
from transformers import AutoTokenizer, AutoModel


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model = AutoModel.from_pretrained("bert-base-cased")
print()
print()

text = "Bayern Munich lost against Vfl Bochum in Bundesliga"

encoded = tokenizer.encode_plus(
    text,
    add_special_tokens=True,
    max_length=512,
    truncation=True,
    return_token_type_ids=False,
    padding='max_length',
    return_attention_mask=True,
    return_tensors="pt"
)

out = model(encoded["input_ids"], encoded["attention_mask"])
print(out[0][:, 0, :].shape)
