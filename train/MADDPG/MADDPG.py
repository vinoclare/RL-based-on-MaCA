#! /usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import os
from common.Replay import Memory


class NetFighterCritic(nn.Module):
    # 输入img(screen)、info信息、所有Agent的action
    # 输出价值q_value

    def __init__(self, agent_num):
        super(NetFighterCritic, self).__init__()
        self.conv = nn.Sequential(  # batch * 100 * 100 * 5
            nn.Conv2d(
                in_channels=5,
                out_channels=8,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.Tanh(),
            nn.MaxPool2d(4),
        )
        self.img_layernorm = nn.LayerNorm(25 * 25 * 8)
        self.info_fc = nn.Sequential(  # batch * 3
            nn.Linear(3, 16),
            nn.LayerNorm(16),
            nn.Tanh(),
        )
        self.action_fc = nn.Sequential(  # batch * (4 * agent_num)
            nn.Linear(4 * agent_num, 64),
            nn.LayerNorm(64),
            nn.Tanh(),
        )
        self.feature_fc = nn.Sequential(  # 25 * 25 * 64 + 256 + 256
            nn.Linear((25 * 25 * 8 + 16 + 64), 64),
            nn.LayerNorm(64),
            nn.Tanh(),
        )
        self.decision_fc = nn.Sequential(
            nn.Linear(64, 1),
        )

    def forward(self, img, info, act):
        img_feature = self.conv(img)
        img_feature = img_feature.view(img_feature.size(0), -1)
        img_feature = self.img_layernorm(img_feature)
        info_feature = self.info_fc(info)
        action_feature = self.action_fc(act)
        combined = torch.cat((img_feature, info_feature.view(info_feature.size(0), -1),
                              action_feature.view(action_feature.size(0), -1)),
                             dim=1)
        feature = self.feature_fc(combined)
        q_value = self.decision_fc(feature)
        return q_value


class NetFighterActor(nn.Module):
    # 决定航向与攻击类型的Actor网络
    # 输入img（screen）信息、info信息、agent存活信息
    # 输出航向（[0-359]离散值）、攻击类型(连续值)

    def __init__(self):
        super(NetFighterActor, self).__init__()
        self.conv = nn.Sequential(  # batch * 100 * 100 * 3
            nn.Conv2d(
                in_channels=5,
                out_channels=8,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.Tanh(),
            nn.MaxPool2d(4),
        )
        self.img_layernorm = nn.LayerNorm(25 * 25 * 8)
        self.info_fc = nn.Sequential(
            nn.Linear(3, 16),
            nn.LayerNorm(16),
            nn.Tanh(),
        )
        self.alive_fc = nn.Sequential(
            nn.Linear(1, 2),
            nn.LayerNorm(2),
            nn.Tanh(),
        )
        self.feature_fc = nn.Sequential(
            nn.Linear((25 * 25 * 8 + 16 + 2), 64),
            nn.LayerNorm(64),
            nn.Tanh(),
        )
        self.decision_fc = nn.Sequential(
            nn.Linear(64, 2),
        )

    def forward(self, img, info, alive):
        img_feature = self.conv(img)
        img_feature = img_feature.view(img_feature.size(0), -1)
        img_feature = self.img_layernorm(img_feature)
        info_feature = self.info_fc(info)
        alive_feature = self.alive_fc(alive)
        combined = torch.cat((img_feature, info_feature.view(info_feature.size(0), -1),
                              alive_feature.view(alive_feature.size(0), -1))
                             , dim=1)
        feature = self.feature_fc(combined)
        decision = self.decision_fc(feature)
        return decision


class RLFighter:
    def __init__(
            self,
            name,
            agent_num,
            attack_num,
            fighter_num,
            actor_lr=1e-5,
            critic_lr=1e-5,
            reward_decay=0.99,
            tau=0.95,
            e_greedy=0.9,
            replace_target_iter=50,
            max_memory_size=1e4,
            batch_size=512,
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
        self.eval_net_critic, self.target_net_critic = NetFighterCritic(self.agent_num), NetFighterCritic(
            self.agent_num)
        self.eval_net_actor, self.target_net_actor = NetFighterActor(), NetFighterActor()

        # 初始化网络参数
        for m1, m2 in zip(self.eval_net_critic.modules(), self.eval_net_actor.modules()):
            # if isinstance(m1, nn.Conv2d) or isinstance(m1, nn.Linear):
            #     init.kaiming_uniform_(m1.weight, mode='fan_in', nonlinearity='leaky_relu')
            if isinstance(m2, nn.Conv2d) or isinstance(m2, nn.Linear):
                init.kaiming_uniform_(m2.weight, mode='fan_in', nonlinearity='leaky_relu')

        self.copy_param(self.eval_net_critic, self.target_net_critic, 0)
        self.copy_param(self.eval_net_actor, self.target_net_actor, 0)

        if self.gpu_enable:
            print('GPU Available!!')
            self.eval_net_critic = self.eval_net_critic.cuda()
            self.target_net_critic = self.target_net_critic.cuda()
            self.eval_net_actor = self.eval_net_actor.cuda()
            self.target_net_actor = self.target_net_actor.cuda()

        self.loss_func = nn.MSELoss()

        # 加载已有的模型参数
        if self.load_state:
            state_dict = torch.load(self.model_path)
            self.eval_net_critic.load_state_dict(state_dict)
            self.target_net_critic.load_state_dict(state_dict)
            self.learn_step_counter = int(self.model_path[-10:-4])

        # 优化器
        # Adam
        self.critic_optimizer = torch.optim.Adam(self.eval_net_critic.parameters(), lr=self.cr_lr)
        self.actor_optimizer = torch.optim.Adam(self.eval_net_actor.parameters(), lr=self.ac_lr)

        # RMSprop
        # self.critic_optimizer = torch.optim.RMSprop(self.eval_net_critic.parameters(), lr=self.lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
        # self.actor_optimizer = torch.optim.RMSprop(self.eval_net_actor.parameters(), lr=self.lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

    def copy_param(self, eval_net, target_net, tau):
        # 将eval网络中的参数复制到target网络中
        # tau: target网络参数保留率
        tem_dict = {}
        for param_tensor in eval_net.state_dict():
            tem_value = tau * target_net.state_dict()[param_tensor] + (1 - tau) * eval_net.state_dict()[param_tensor]
            tem_dict[param_tensor] = tem_value
        target_net.load_state_dict(tem_dict)

    def store_replay(self, s, alive, alive_, a, ma, oa, r, s_):
        # 将每一step的经验加入经验池
        self.memory.store_replay(s, alive, alive_, a, ma, oa, r, s_)

    def get_memory_size(self):
        return self.memory.get_size()

    # 选择行为(只针对单个数据，不是batch，用于决策时)
    def choose_action(self, img_obs, info_obs, alive):
        img_obs = torch.unsqueeze(torch.FloatTensor(img_obs), 0)
        info_obs = torch.unsqueeze(torch.FloatTensor(info_obs), 0)
        alive = torch.FloatTensor(alive)
        alive = torch.unsqueeze(alive, 0)
        if self.gpu_enable:
            img_obs = img_obs.cuda()
            info_obs = info_obs.cuda()
            alive = alive.cuda()
        act = self.target_net_actor(img_obs, info_obs, alive).cpu()
        course = torch.tanh(act[:, 0]) * 180
        try:
            course = int(course)
            course = int(course + self.noise_rate * np.random.randn())  # 对航向添加一个随机扰动
            attack = act[:, 1]  # 攻击
            attack = int(1 / (1 + torch.exp(-attack)) * self.attack_num)
        except:
            course = 179
            attack = 0
        return [course, 1, 0, attack]

    # 针对batch的数据选择行为(使用eval_actor网络计算，用于训练)
    def choose_action_batch(self, img_obs_batch, info_obs_batch, alive_batch):
        # noise = torch.randn(self.batch_size, 1) * self.noise_rate
        if self.gpu_enable:
            img_obs_batch = img_obs_batch.cuda()
            info_obs_batch = info_obs_batch.cuda()
            alive_batch = alive_batch.cuda()
            # noise = noise.cuda()
        act = self.eval_net_actor(img_obs_batch, info_obs_batch, alive_batch)  # 航向
        course = torch.tanh(act[:, 0]) * 180
        attack = act[:, 1]
        attack = 1 / (1 + torch.exp(-attack)) * self.attack_num
        act1 = torch.ones([course.size(0), 1]).cuda()
        act2 = torch.zeros([course.size(0), 1]).cuda()
        # course = course + noise
        course = torch.unsqueeze(course, 1)
        attack = torch.unsqueeze(attack, 1)
        action = torch.cat([course, act1, act2, attack], 1)
        return action

    # 针对batch的数据选择行为(使用target_actor网络计算)
    def choose_action_batch_tar(self, img_obs_batch, info_obs_batch, alive_batch):
        noise = torch.randn(self.batch_size, 1) * self.noise_rate
        if self.gpu_enable:
            img_obs_batch = img_obs_batch.cuda()
            info_obs_batch = info_obs_batch.cuda()
            alive_batch = alive_batch.cuda()
            noise = noise.cuda()
        act = self.target_net_actor(img_obs_batch, info_obs_batch, alive_batch)  # 航向+攻击
        course = torch.tanh(act[:, 0]) * 180
        attack = act[:, 1]
        attack = 1 / (1 + torch.exp(-attack)) * self.attack_num
        act1 = torch.ones([course.size(0), 1]).cuda()
        act2 = torch.zeros([course.size(0), 1]).cuda()
        # course = course + noise
        course = torch.unsqueeze(course, 1)
        attack = torch.unsqueeze(attack, 1)
        action = torch.cat([course, act1, act2, attack], 1)
        return action

    def learn(self, save_path, writer):
        # 复制参数+保存参数
        # learn50次复制/保存一次参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.copy_param(self.eval_net_critic, self.target_net_critic, self.tau)
            self.copy_param(self.eval_net_actor, self.target_net_actor, self.tau)
            print('\ntarget_params_replaced\n')
            step_counter_str = '%09d' % self.learn_step_counter
            torch.save(self.target_net_critic.state_dict(),
                       save_path + '/critic/model_' + step_counter_str + '.pkl')
            torch.save(self.target_net_actor.state_dict(),
                       save_path + '/actor/model_' + step_counter_str + '.pkl')
            if self.pkl_counter < self.max_pkl:
                self.pkl_counter += 1
            else:
                # 删除最旧的模型参数
                critic_path = os.path.join(save_path, 'Critic.py')
                actor_path = os.path.join(save_path, 'actor')

                files = os.listdir(critic_path)
                for file in files:
                    if file.endswith('pkl'):
                        os.remove(os.path.join(critic_path, file))
                        break

                files = os.listdir(actor_path)
                for file in files:
                    if file.endswith('pkl'):
                        os.remove(os.path.join(actor_path, file))
                        break

        # 采样Replay
        [s_screen_batch, s_info_batch, alive_batch, alive__batch, self_a_batch,
         mate_a_batch, other_a_batch, r_batch, s__screen_batch, s__info_batch] = self.memory.sample_replay(
            self.batch_size, self.gpu_enable)

        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()

        # 计算agent自身的action
        self.action = self.choose_action_batch(s_screen_batch, s_info_batch, alive_batch)
        self.action = self.action.view(self.action.size(0), 1, self.action.size(1))
        self.action_ = self.choose_action_batch_tar(s__screen_batch, s__info_batch, alive__batch)
        self.action_ = self.action_.view(self.action_.size(0), 1, self.action_.size(1))
        # 双方全部的action
        self_a_batch = torch.unsqueeze(self_a_batch, 1)
        memory_action = torch.cat([self_a_batch, mate_a_batch, other_a_batch], 1).view(mate_a_batch.size(0), -1)
        all_action = torch.cat([self.action, mate_a_batch, other_a_batch], 1).view(mate_a_batch.size(0), -1)
        all__action = torch.cat([self.action_, mate_a_batch, other_a_batch], 1).view(mate_a_batch.size(0), -1)
        if self.gpu_enable:
            all_action = all_action.cuda()
        all_action = all_action.float()
        q_eval = self.eval_net_critic(s_screen_batch, s_info_batch, memory_action)
        q_next = self.target_net_critic(s__screen_batch, s__info_batch, all__action).detach()  # detach使tensor不能反向传播
        q_target = r_batch + self.gamma * q_next.view(-1, 1)  # shape (batch, 1)

        # 反向传播、优化（有点问题）
        # critic
        critic_loss = self.loss_func(q_eval, q_target)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.eval_net_critic.parameters(), max_norm=5, norm_type=2)
        self.critic_optimizer.step()

        # actor
        q = self.eval_net_critic(s_screen_batch, s_info_batch, all_action)
        actor_loss = -torch.mean(q)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.eval_net_actor.parameters(), max_norm=5, norm_type=2)
        self.actor_optimizer.step()

        self.cost_critic_his.append(critic_loss)
        self.cost_actor_his.append(actor_loss)
        print("learn_step: %d, %s_critic_loss: %.4f, %s_actor_loss: %.4f, mean_reward: %.2f" % (
            self.learn_step_counter, self.name, critic_loss, self.name, actor_loss, torch.mean(r_batch)))

        # 训练过程保存
        writer.add_scalar(tag='%s_actor_loss' % self.name, scalar_value=actor_loss, global_step=self.learn_step_counter)
        writer.add_scalar(tag='%s_critic_loss' % self.name, scalar_value=critic_loss, global_step=self.learn_step_counter)

        self.learn_step_counter += 1


class NetDetector(nn.Module):
    def __init__(self, n_actions):
        super(NetDetector, self).__init__()
        self.conv1 = nn.Sequential(  # 100 * 100 * 3
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(  # 50 * 50 * 16
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 25 * 25 * 32
        )
        self.info_fc = nn.Sequential(
            nn.Linear(3, 256),
            nn.Tanh(),
        )
        self.feature_fc = nn.Sequential(  # 25 * 25 * 32 + 256
            nn.Linear((25 * 25 * 32 + 256), 512),
            nn.ReLU(),
        )
        self.decision_fc = nn.Linear(512, n_actions)

    def forward(self, img, info):
        img_feature = self.conv1(img)
        img_feature = self.conv2(img_feature)
        info_feature = self.info_fc(info)
        combined = torch.cat((img_feature.view(img_feature.size(0), -1), info_feature.view(info_feature.size(0), -1)),
                             dim=1)
        feature = self.feature_fc(combined)
        action = self.decision_fc(feature)
        return action
