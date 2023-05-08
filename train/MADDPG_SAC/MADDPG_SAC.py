#! /usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import os
from common.Replay2 import Memory
from train.MADDPG_SAC.Critic_SAC import NetFighterCritic
from train.MADDPG_SAC.Actor_SAC import NetFighterActor
from torch.distributions import Normal


class RLFighter:
    def __init__(
            self,
            name,
            agent_num,
            attack_num,
            fighter_num,
            radar_num,
            noise_rate=5,
            actor_lr=3e-4,
            critic_lr=3e-4,
            q_lr=3e-4,
            reward_decay=0.99,
            tau=0.99,
            reward_scaling=0.3,
            alpha=0.2,
            replace_target_iter=50,
            max_memory_size=1e4,
            batch_size=512,
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
        self.alpha = alpha  # 温度系数
        self.batch_size = batch_size
        self.reward_scaling = reward_scaling  # reward缩放系数
        self.noise_rate = noise_rate  # 噪声率（用于为动作添加随机扰动）
        self.ac_lr = actor_lr  # actor网络学习率
        self.cr_lr = critic_lr  # critic网络学习率
        self.gamma = reward_decay  # 奖励的衰减率
        self.tau = tau  # 模型参数复制时的参数保留率
        self.replace_target_iter = replace_target_iter  # eval网络参数更新到target网络的频率

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

        # 初始化网络
        self.eval_net_critic_fighter, self.target_net_critic_fighter = NetFighterCritic(
            self.agent_num), NetFighterCritic(self.agent_num)
        self.policy_net_fighter = NetFighterActor()

        # 初始化网络参数
        # for m1 in self.policy_net_fighter.modules():
        #     if isinstance(m1, nn.Conv2d) or isinstance(m1, nn.Linear):
        #         init.kaiming_uniform_(m1.weight, mode='fan_in', nonlinearity='leaky_relu')

        self.copy_param(self.eval_net_critic_fighter, self.target_net_critic_fighter, 0)

        if self.gpu_enable:
            print('GPU Available!!')
            self.eval_net_critic_fighter = self.eval_net_critic_fighter.cuda()
            self.target_net_critic_fighter = self.target_net_critic_fighter.cuda()
            self.policy_net_fighter = self.policy_net_fighter.cuda()

        self.loss_func = nn.MSELoss()

        # 加载已有的模型参数
        # if self.load_state:
        #     state_dict = torch.load(self.model_path)
        #     self.eval_net_critic_fighter.load_state_dict(state_dict)
        #     self.target_net_critic_fighter.load_state_dict(state_dict)
        #     self.learn_step_counter = int(self.model_path[-10:-4])

        # 优化器
        # Adam
        self.critic_optimizer_fighter = torch.optim.Adam(self.eval_net_critic_fighter.parameters(), lr=self.cr_lr)
        self.policy_optimizer_fighter = torch.optim.Adam(self.policy_net_fighter.parameters(), lr=self.ac_lr)

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

    def save_best_model(self, save_path, epoch_reward):
        torch.save(self.target_net_critic_fighter.state_dict(),
                   save_path + '/critic/model_' + str(int(epoch_reward)) + '.pkl')
        torch.save(self.policy_net_fighter.state_dict(),
                   save_path + '/actor/model_' + str(int(epoch_reward)) + '.pkl')
        # 超过5个模型后删除最低reward的
        critic_path = save_path + '/critic'
        actor_path = save_path + '/actor'
        critic_pkls = os.listdir(critic_path)
        actor_pkls = os.listdir(actor_path)
        if len(critic_pkls) > 5:
            rewards = []
            for file in critic_pkls:
                reward = int(float(file[6:-4]))
                rewards.append(reward)
            index = np.argmin(rewards)
            os.remove(os.path.join(critic_path, critic_pkls[index]))
            os.remove(os.path.join(actor_path, actor_pkls[index]))

    # 选择行为(只针对单个数据，不是batch，用于决策时)
    def choose_action(self, img_obs, info_obs):
        img_obs = torch.unsqueeze(torch.FloatTensor(img_obs), 0)
        info_obs = torch.unsqueeze(torch.FloatTensor(info_obs), 0)
        if self.gpu_enable:
            img_obs = img_obs.cuda()
            info_obs = info_obs.cuda()

        means, log_stds = self.policy_net_fighter(img_obs, info_obs)
        means = means.cpu()
        log_stds = log_stds.cpu()
        stds = torch.exp(log_stds)
        dist = Normal(means, stds)
        act = dist.rsample()

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
            # noise = noise.cuda()

        means, log_stds = self.policy_net_fighter(img_obs_batch, info_obs_batch)
        stds = torch.exp(log_stds)
        dist = Normal(means, stds)
        act = dist.rsample()

        act2 = torch.tanh(act)
        log_prob = dist.log_prob(act) - torch.log(1 - act2.pow(2) + torch.tensor(1e-6).float())
        log_prob = log_prob.sum(dim=1)
        log_prob = torch.unsqueeze(log_prob, 1)
        return act2, log_prob

    def save_to_file(self, path):
        self.memory.save_to_file(path)

    def load_from_file(self, path):
        self.memory.load_from_file(path)

    def learn(self, writer, batch_indexes, mate_agents, other_a_batch):
        # 复制参数+保存参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.copy_param(self.eval_net_critic_fighter, self.target_net_critic_fighter, self.tau)

        # 采样Replay
        [s_screen_batch, s_info_batch, alive_batch, self_a_batch,
         r_batch, s__screen_batch, s__info_batch, done_batch] = self.memory.sample_replay(batch_indexes)

        # 队友action
        for i in range(9):
            [_, _, _, action, _, _, _, _] = mate_agents[i].memory.sample_replay(batch_indexes)
            if i == 0:
                mate_a_batch = torch.unsqueeze(action, 1)
            else:
                tem_act = torch.unsqueeze(action, 1)
                mate_a_batch = torch.cat([mate_a_batch, tem_act], 1)

        # 自己的action
        self_a_batch = torch.unsqueeze(self_a_batch, 1)

        # 反向传播、优化
        # Critic
        with torch.no_grad():
            action_, log_probs_ = self.choose_action_batch(s__screen_batch, s__info_batch)
            action_ = torch.unsqueeze(action_, 1)
            all_action_ = torch.cat([action_, mate_a_batch, other_a_batch], 1).view(mate_a_batch.size(0), -1)
            all_mem_action = torch.cat([self_a_batch, mate_a_batch, other_a_batch], 1).view(mate_a_batch.size(0), -1)
            q1_next, q2_next = self.target_net_critic_fighter(s__screen_batch, s__info_batch, all_action_)
            q_target = torch.min(q1_next, q2_next) - self.alpha * log_probs_
            q_target = self.reward_scaling * r_batch + alive_batch * self.gamma * q_target * (1 - done_batch)
        q1_cur, q2_cur = self.eval_net_critic_fighter(s_screen_batch, s_info_batch, all_mem_action)

        critic_loss = self.loss_func(q1_cur, q_target) + self.loss_func(q2_cur, q_target)
        self.critic_optimizer_fighter.zero_grad()
        critic_loss.backward()
        self.critic_optimizer_fighter.step()

        # Actor
        action, log_probs = self.choose_action_batch(s_screen_batch, s_info_batch)
        action = torch.unsqueeze(action, 1)
        all_action = torch.cat([action, mate_a_batch, other_a_batch], 1).view(mate_a_batch.size(0), -1)
        q1_pi, q2_pi = self.eval_net_critic_fighter(s_screen_batch, s_info_batch, all_action)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = ((self.alpha * log_probs) - min_q_pi).mean()
        self.policy_optimizer_fighter.zero_grad()
        actor_loss.backward()
        # nn.utils.clip_grad_norm_(self.eval_net_critic_fighter.parameters(), 0.5)
        self.policy_optimizer_fighter.step()

        self.learn_step_counter += 1
        return actor_loss, critic_loss
