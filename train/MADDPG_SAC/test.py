import sys
sys.path.append('D:/MaCA/')

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import os
from common.Replay2 import Memory
from train.MADDPG_SAC.Critic import NetFighterCritic
from train.MADDPG_SAC.Actor import NetFighterActor
from train.MADDPG_SAC.Q import NetFighterQ

model = NetFighterCritic(10)
print(model)