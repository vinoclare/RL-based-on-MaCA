from train.MAAC.Critic import NetFighterCritic as FC
import torch

img = torch.rand([10, 5, 100, 100])
info = torch.rand([10, 3])
action = torch.rand([10, 40])

model = FC(10)
res = model(img, info, action)
print('\n', res)
