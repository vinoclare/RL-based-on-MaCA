from main import seed_everything
import torch
import numpy as np
from Actor import NetFighterActor

seed_everything(1029)
model = NetFighterActor()
input1 = torch.rand(1, 2, 100, 100)
input2 = torch.rand(1, 3)
out = model(input1, input2)
print(out)