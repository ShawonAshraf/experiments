import torch
import torch.nn.functional as F

preds = torch.ones(300, )
preds[:37] = torch.arange(37)
actual = preds.clone()


non_padded = (actual != 1).nonzero()

matches = torch.eq(preds[non_padded], actual[non_padded])
print(matches.sum() / actual[non_padded].size()[0])
