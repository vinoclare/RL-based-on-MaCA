from collections import *
import numpy as np
import torch

"""
红色方经验池类
"""


class Replay(object):
    def __init__(self, act):
        self.s_act = act  # 自身的action

    def get_s_act(self):
        return self.s_act


class Memory(object):
    def __init__(self, max_memory_size):
        self.data = deque()
        self.memory_counter = 0  # 当前memory大小
        self.max_memory_size = max_memory_size  # memory容量

    def store_replay(self, sa):
        # 将每一step的经验加入经验池
        # 经验池未满时正常加入经验
        replay = Replay(sa)
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
        for i in indexes:
            replay = self.data[i]
            self_a_batch.append(replay.get_s_act())
        self_a_batch = torch.FloatTensor(np.array(self_a_batch)).cuda()

        return self_a_batch

    def save_to_file(self, path):
        np.save(path, self.data)

    def load_from_file(self, path):
        self.data = np.load(path, allow_pickle=True)
        self.data = deque(self.data)
        self.memory_counter = len(self.data)
