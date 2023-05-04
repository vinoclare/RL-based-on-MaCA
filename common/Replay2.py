from collections import *
import random

import numpy as np
import torch

"""
经验池类
"""


class Replay(object):
    def __init__(self, s_screen, s_info, alive, self_act,
                 r, s__screen, s__info, done):
        self.s_screen = s_screen  # 当前screen信息
        self.s_info = s_info  # 当前info信息
        self.alive = alive  # 当前时刻存活信息
        self.s_act = self_act  # 自身的action
        self.r = r  # reward
        self.s__screen = s__screen  # 下一步screen信息
        self.s__info = s__info  # 下一步info信息
        self.done = done  # epoch是否完成

    def get_s_screen(self):
        return self.s__screen

    def get_s_info(self):
        return self.s_info

    def get_alive(self):
        return self.alive

    def get_s_act(self):
        return self.s_act

    def get_r(self):
        return self.r

    def get_s__screen(self):
        return self.s__screen

    def get_s__info(self):
        return self.s__info

    def get_done(self):
        return self.done


class Memory(object):
    def __init__(self, max_memory_size):
        self.data = deque()
        self.memory_counter = 0  # 当前memory大小
        self.max_memory_size = max_memory_size  # memory容量

    def store_replay(self, s, alive, sa, r, s_, d):
        # 将每一step的经验加入经验池
        # 经验池未满时正常加入经验
        replay = Replay(s['screen'], s['info'], alive, sa, r, s_['screen'], s_['info'], d)
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
        s_screen_batch = []
        s_info_batch = []
        alive_batch = []
        self_a_batch = []
        r_batch = []
        s__screen_batch = []
        s__info_batch = []
        done_batch = []
        for i in indexes:
            replay = self.data[i]
            s_screen_batch.append(replay.get_s_screen())
            s_info_batch.append(replay.get_s_info())
            alive_batch.append(replay.get_alive())
            self_a_batch.append(replay.get_s_act())
            r_batch.append(replay.get_r())
            s__screen_batch.append(replay.get_s__screen())
            s__info_batch.append(replay.get_s__info())
            done_batch.append(replay.get_done())

        s_screen_batch = torch.FloatTensor(np.array(s_screen_batch)).cuda()
        s_info_batch = torch.FloatTensor(np.array(s_info_batch)).cuda()
        alive_batch = torch.FloatTensor(np.array(alive_batch)).cuda()
        self_a_batch = torch.FloatTensor(np.array(self_a_batch)).cuda()
        r_batch = torch.FloatTensor(np.array(r_batch)).cuda()
        s__screen_batch = torch.FloatTensor(np.array(s__screen_batch)).cuda()
        s__info_batch = torch.FloatTensor(np.array(s__info_batch)).cuda()
        done_batch = torch.FloatTensor(np.array(done_batch)).cuda()
        alive_batch = alive_batch.view(-1, 1)
        r_batch = r_batch.view(-1, 1)
        done_batch = done_batch.view(-1, 1)

        return [s_screen_batch, s_info_batch, alive_batch, self_a_batch,
                r_batch, s__screen_batch, s__info_batch, done_batch]

    def save_to_file(self, path):
        np.save(path, self.data)

    def load_from_file(self, path):
        self.data = np.load(path, allow_pickle=True)
        self.data = deque(self.data)
        self.memory_counter = len(self.data)
