from collections import *
import random

import numpy as np
import torch

"""
红色方经验池类
"""


class Replay(object):
    def __init__(self, act, pos):
        self.s_act = act  # 自身的action
        self.pos = pos  # 自身的位置

    def get_s_act(self):
        return self.s_act

    def get_pos(self):
        return self.pos


class Memory(object):
    def __init__(self, max_memory_size):
        self.data = deque()
        self.memory_counter = 0  # 当前memory大小
        self.max_memory_size = max_memory_size  # memory容量

    def store_replay(self, sa, poses):
        # 将每一step的经验加入经验池
        # 经验池未满时正常加入经验
        replay = Replay(sa, poses)
        if self.memory_counter < self.max_memory_size:
            self.data.append(replay)
            self.memory_counter += 1
        else:  # 经验池已满，则删除最旧的经验，添加新经验
            # 删除
            self.data.popleft()
            # 添加
            self.data.append(replay)

    def get_size(self):
        # 获取经验池大小
        return self.memory_counter

    def __clear_memory(self):
        # 清空经验池
        self.data.clear()

    def sample_replay(self, indexes):
        # 经验池采样
        self_a_batch = []
        pos_batch = []
        for i in indexes:
            replay = self.data[i]
            self_a_batch.append(replay.get_s_act())
            pos_batch.append(replay.get_pos())
        self_a_batch = torch.FloatTensor(np.array(self_a_batch)).cuda()
        pos_batch = torch.FloatTensor(np.array(pos_batch)).cuda()

        return self_a_batch, pos_batch
