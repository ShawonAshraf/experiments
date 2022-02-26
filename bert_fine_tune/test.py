import torch
import torch.nn as nn


x = torch.randn((2, 5, 1))
print(x)
softmax = nn.Softmax(dim=1)

s = softmax(x)
print(s)

out, _ = torch.max(s, dim=1)
print(out)
