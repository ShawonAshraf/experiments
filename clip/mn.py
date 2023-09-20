# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# %%
from torchvision.datasets import MNIST

dataset = MNIST(root="./dataset", download=True)

# %%
label_prompts = [
    f"This is a handwritten image of the digit {i}."
    for i in range(10)
]

label_prompts_tokenised = torch.stack([clip.tokenize(lp) for lp in label_prompts], dim=0)
label_prompts_tokenised.size()

label_prompts_tokenised = label_prompts_tokenised.to(device)

# %%
from tqdm.auto import tqdm

images = torch.stack([preprocess(dataset[i][0]) for i in range(len(dataset))], dim=0)
images = images.to(device)

# %%
images.size()

# %%
labels = torch.tensor([dataset[i][1] for i in range(len(dataset))])
labels.size()

# %%
predicted = []

with torch.no_grad():
    image_features = model.encode_image(images)
    image_features = image_features.to(device)
    
    text_features = model.encode_text(label_prompts_tokenised)
    text_features = text_features.to(device)
    
    logits_per_image, _ = model(image_features, text_features)
    probs = logits_per_image.softmax(dim=-1).cpu()
    predicted.append(torch.argmax(probs, dim=-1))

# %%



