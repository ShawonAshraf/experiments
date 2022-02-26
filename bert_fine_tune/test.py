import torch
import torch.nn as nn

# batch size, hidden_last, out
x = torch.randn((2, 5, 1))
print("x")
print(x)
print()

softmax = nn.Softmax(dim=1)
sigmoid = nn.Sigmoid()

s = sigmoid(x)
# s = softmax(x)
print("activation")
print(s)
print()

out, _ = torch.max(s, dim=1)
print("torch.max")
print(out)
print()
