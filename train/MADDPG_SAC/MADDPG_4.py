#! /usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import os
from common.Replay2 import Memory
from train.MADDPG.Critic import NetFighterCritic
from train.MADDPG.Actor import NetFighterActor


class RLFighter:
    def __init__(
            self,
            name,
            agent_num,
            attack_num,
            fighter_num,
            radar_num,
            actor_lr=1e-5,
            critic_lr=1e-6,
            reward_decay=0.99,
            tau=0.95,
            e_greedy=0.9,
            replace_target_iter=50,
            max_memory_size=1e4,
            batch_size=256,
            e_greedy_increment=None,
            output_graph=False,
            load_state=False,
            model_path='',
    ):
        self.name = name  # agent阵营 blue or red
        self.action = []  # agent自身一个经验batch内的行动
        self.action_ = []  # agent下一个时刻的行动（batch）
        self.agent_num = agent_num  # 双方agent总数量
        self.attack_num = attack_num  # 总攻击类型数量
        self.fighter_num = fighter_num  # 本队fighter数量
        self.radar_num = radar_num  # 雷达频点数
        self.batch_size = batch_size
        self.noise_rate = 5  # 噪声率（用于为动作添加随机扰动）
        self.ac_lr = actor_lr  # actor网络学习率
        self.cr_lr = critic_lr  # critic网络学习率
        self.gamma = reward_decay  # 奖励的衰减率
        self.tau = tau  # 模型参数复制时的参数保留率
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter  # eval网络参数更新到target网络的频率
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # 经验池
        self.memory = Memory(max_memory_size)

        # 模型加载保存
        self.load_state = load_state
        self.model_path = model_path
        self.pkl_counter = 0
        self.max_pkl = 10
        self.gpu_enable = torch.cuda.is_available()

        # 总训练步数
        self.learn_step_counter = 1

        # 损失记录
        self.cost_critic_his = []
        self.cost_actor_his = []

        # 初始化网络
        self.eval_net_critic_fighter, self.target_net_critic_fighter = NetFighterCritic(self.agent_num), NetFighterCritic(self.agent_num)
        self.eval_net_actor_fighter, self.target_net_actor_fighter = NetFighterActor(), NetFighterActor()

        # 初始化网络参数
        for m1 in self.eval_net_actor_fighter.modules():
            if isinstance(m1, nn.Conv2d) or isinstance(m1, nn.Linear):
                init.kaiming_uniform_(m1.weight, mode='fan_in', nonlinearity='leaky_relu')

        self.copy_param(self.eval_net_critic_fighter, self.target_net_critic_fighter, 0)
        self.copy_param(self.eval_net_actor_fighter, self.target_net_actor_fighter, 0)

        if self.gpu_enable:
            print('GPU Available!!')
            self.eval_net_critic_fighter = self.eval_net_critic_fighter.cuda()
            self.target_net_critic_fighter = self.target_net_critic_fighter.cuda()
            self.eval_net_actor_fighter = self.eval_net_actor_fighter.cuda()
            self.target_net_actor_fighter = self.target_net_actor_fighter.cuda()

        self.loss_func = nn.MSELoss()

        # 加载已有的模型参数
        if self.load_state:
            state_dict = torch.load(self.model_path)
            self.eval_net_critic_fighter.load_state_dict(state_dict)
            self.target_net_critic_fighter.load_state_dict(state_dict)
            self.learn_step_counter = int(self.model_path[-10:-4])

        # 优化器
        # Adam
        self.critic_optimizer_fighter = torch.optim.Adam(self.eval_net_critic_fighter.parameters(), lr=self.cr_lr)
        self.actor_optimizer_fighter = torch.optim.Adam(self.eval_net_actor_fighter.parameters(), lr=self.ac_lr)

    def copy_param(self, eval_net, target_net, tau):
        # 将eval网络中的参数复制到target网络中
        # tau: target网络参数保留率
        tem_dict = {}
        for param_tensor in eval_net.state_dict():
            tem_value = tau * target_net.state_dict()[param_tensor] + (1 - tau) * eval_net.state_dict()[param_tensor]
            tem_dict[param_tensor] = tem_value
        target_net.load_state_dict(tem_dict)

    def store_replay(self, s, alive, a, r, s_, d):
        # 将每一step的经验加入经验池
        self.memory.store_replay(s, alive, a, r, s_, d)

    def get_memory_size(self):
        return self.memory.get_size()

    # 选择行为(只针对单个数据，不是batch，用于决策时)
    def choose_action(self, img_obs, info_obs):
        img_obs = torch.unsqueeze(torch.FloatTensor(img_obs), 0)
        info_obs = torch.unsqueeze(torch.FloatTensor(info_obs), 0)
        if self.gpu_enable:
            img_obs = img_obs.cuda()
            info_obs = info_obs.cuda()
        
        act = self.target_net_actor_fighter(img_obs, info_obs).cpu()
        action = torch.tanh(act)
        action = (action + 1) / 2
        course = int((action[:, 0] * 2 - 1) * 180 + self.noise_rate * np.random.randn())
        radar = int(action[:, 1] * (self.radar_num + 1))
        disturb = int(action[:, 2] * (self.radar_num + 2))
        attack = int(action[:, 3] * self.attack_num)
        return [course, radar, disturb, attack], action[0].detach().numpy()

    # 针对batch的数据选择行为(使用eval_actor网络计算，用于训练)
    def choose_action_batch(self, img_obs_batch, info_obs_batch):
        # noise = torch.randn(self.batch_size, 1) * self.noise_rate
        if self.gpu_enable:
            img_obs_batch = img_obs_batch.cuda()
            info_obs_batch = info_obs_batch.cuda()

        act = self.eval_net_actor_fighter(img_obs_batch, info_obs_batch)
        action = torch.tanh(act)
        return action

    # 针对batch的数据选择行为(使用target_actor网络计算)
    def choose_action_batch_tar(self, img_obs_batch, info_obs_batch):
        if self.gpu_enable:
            img_obs_batch = img_obs_batch.cuda()
            info_obs_batch = info_obs_batch.cuda()

        act = self.target_net_actor_fighter(img_obs_batch, info_obs_batch)
        action = torch.tanh(act)
        return action

    def save_to_file(self, path):
        self.memory.save_to_file(path)

    def load_from_file(self, path):
        self.memory.load_from_file(path)

    def learn(self, writer, batch_indexes, mate_agents, other_a_batch):
        # 复制参数+保存参数
        # learn50次复制/保存一次参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.copy_param(self.eval_net_critic_fighter, self.target_net_critic_fighter, self.tau)
            self.copy_param(self.eval_net_actor_fighter, self.target_net_actor_fighter, self.tau)
            # print('\ntarget_params_replaced\n')
            # step_counter_str = '%09d' % self.learn_step_counter
            # critic_path = save_path + '/critic/'
            # actor_path = save_path + '/actor/'
            # if not os.path.exists(critic_path):
            #     os.mkdir(critic_path)
            # if not os.path.exists(actor_path):
            #     os.mkdir(actor_path)
            # torch.save(self.target_net_critic_fighter.state_dict(),
            #            save_path + '/critic/model_' + step_counter_str + '.pkl')
            # torch.save(self.target_net_actor_fighter.state_dict(),
            #            save_path + '/actor/model_' + step_counter_str + '.pkl')
            # if self.pkl_counter < self.max_pkl:
            #     self.pkl_counter += 1
            # else:
            #     # 删除最旧的模型参数
            #     files = os.listdir(critic_path)
            #     for file in files:
            #         if file.endswith('pkl'):
            #             os.remove(os.path.join(critic_path, file))
            #             break
            #
            #     files = os.listdir(actor_path)
            #     for file in files:
            #         if file.endswith('pkl'):
            #             os.remove(os.path.join(actor_path, file))
            #             break

        # 采样Replay
        [s_screen_batch, s_info_batch, alive_batch, self_a_batch,
         r_batch, s__screen_batch, s__info_batch, done_batch] = self.memory.sample_replay(batch_indexes)

        self.critic_optimizer_fighter.zero_grad()
        self.actor_optimizer_fighter.zero_grad()

        # 计算agent自身的action
        self.action = self.choose_action_batch(s_screen_batch, s_info_batch)
        self.action = self.action.view(self.action.size(0), 1, self.action.size(1))
        self.action_ = self.choose_action_batch_tar(s__screen_batch, s__info_batch)
        self.action_ = self.action_.view(self.action_.size(0), 1, self.action_.size(1))

        # 队友action
        for i in range(9):
            [_, _, _, action, _, _, _, _] = mate_agents[i].memory.sample_replay(batch_indexes)
            if i == 0:
                mate_a_batch = torch.unsqueeze(action, 1)
            else:
                tem_act = torch.unsqueeze(action, 1)
                mate_a_batch = torch.cat([mate_a_batch, tem_act], 1)

        # 双方全部的action
        self_a_batch = torch.unsqueeze(self_a_batch, 1)
        memory_action = torch.cat([self_a_batch, mate_a_batch, other_a_batch], 1).view(mate_a_batch.size(0), -1)
        all_action = torch.cat([self.action, mate_a_batch, other_a_batch], 1).view(mate_a_batch.size(0), -1)
        all__action = torch.cat([self.action_, mate_a_batch, other_a_batch], 1).view(mate_a_batch.size(0), -1)
        if self.gpu_enable:
            all_action = all_action.cuda()
        all_action = all_action.float()
        q_eval = self.eval_net_critic_fighter(s_screen_batch, s_info_batch, memory_action)
        q_next = self.target_net_critic_fighter(s__screen_batch, s__info_batch, all__action).detach()  # detach使tensor不能反向传播
        q_target = r_batch + self.gamma * q_next.view(-1, 1) * alive_batch * (1 - done_batch)  # shape (batch, 1)

        # 反向传播、优化
        # critic
        critic_loss = self.loss_func(q_eval, q_target)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.eval_net_critic_fighter.parameters(), max_norm=5, norm_type=2)
        self.critic_optimizer_fighter.step()

        # actor
        q = self.eval_net_critic_fighter(s_screen_batch, s_info_batch, all_action)
        actor_loss = -torch.mean(q)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.eval_net_actor_fighter.parameters(), max_norm=5, norm_type=2)
        self.actor_optimizer_fighter.step()

        self.cost_critic_his.append(critic_loss)
        self.cost_actor_his.append(actor_loss)

        self.learn_step_counter += 1
        return actor_loss, critic_loss
