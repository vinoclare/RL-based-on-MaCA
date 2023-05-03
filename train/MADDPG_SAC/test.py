from Attention import Attention
import torch


model = Attention()
encodings = [torch.rand(1, 32 * 4 * 4 + 64 + 128) for i in range(10)]
out = model(encodings)
print(out.shape)