#! /usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.init as init
import os
from common.Replay import Memory
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter


class NetFighterValue(nn.Module):
    # 输入img(screen)、info信息
    # 输出价值value

    def __init__(self, init_w=3e-3):
        super(NetFighterValue, self).__init__()
        self.conv = nn.Sequential(  # batch * 100 * 100 * 5
            nn.Conv2d(
                in_channels=5,
                out_channels=8,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(4),
        )
        self.img_layernorm = nn.LayerNorm(25 * 25 * 8)
        self.info_fc = nn.Sequential(  # batch * 3
            nn.Linear(3, 64),
            nn.LayerNorm(64),
            nn.Tanh(),
        )
        self.feature_fc = nn.Sequential(
            nn.Linear((25 * 25 * 8 + 64), 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        self.decision_fc = nn.Linear(256, 1)
        self.decision_fc.weight.data.uniform_(-init_w, init_w)
        self.decision_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, img, info):
        img_feature = self.conv(img)
        img_feature = img_feature.view(img_feature.size(0), -1)
        img_feature = self.img_layernorm(img_feature)
        info_feature = self.info_fc(info)
        combined = torch.cat((img_feature, info_feature.view(info_feature.size(0), -1)),
                             dim=1)
        feature = self.feature_fc(combined)
        value = self.decision_fc(feature)
        return value


class NetFighterSQ(nn.Module):
    # 输入img(screen)、info信息、所有Agent的action
    # 输出价值value

    def __init__(self, agent_num, init_w=3e-3):
        super(NetFighterSQ, self).__init__()
        self.conv = nn.Sequential(  # batch * 100 * 100 * 5
            nn.Conv2d(
                in_channels=5,
                out_channels=8,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(4),
        )
        self.img_layernorm = nn.LayerNorm(25 * 25 * 8)
        self.info_fc = nn.Sequential(  # batch * 3
            nn.Linear(3, 64),
            nn.LayerNorm(64),
            nn.Tanh(),
        )
        self.action_fc = nn.Sequential(  # batch * (4 * agent_num)
            nn.Linear(4 * agent_num, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
        )
        self.feature_fc = nn.Sequential(  # 25 * 25 * 64 + 256 + 256
            nn.Linear((25 * 25 * 8 + 64 + 128), 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        self.decision_fc = nn.Linear(256, 1)
        self.decision_fc.weight.data.uniform_(-init_w, init_w)
        self.decision_fc.bias.data.uniform_(-init_w, init_w)

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


class NetFighterPolicy(nn.Module):
    # 决定航向与攻击类型的Policy网络
    # 输入img（screen）信息、info信息、agent存活信息
    # 输出航向（[0-359]离散值）、攻击类型(连续值)

    def __init__(self, radar_num, attack_num, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(NetFighterPolicy, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.radar_num = radar_num
        self.attack_num = attack_num

        self.conv = nn.Sequential(  # batch * 100 * 100 * 3
            nn.Conv2d(
                in_channels=5,
                out_channels=8,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(4),
        )
        self.img_layernorm = nn.LayerNorm(25 * 25 * 8)
        self.info_fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.LayerNorm(64),
            nn.Tanh(),
        )
        self.alive_fc = nn.Sequential(
            nn.Linear(1, 4),
            nn.LayerNorm(4),
            nn.Tanh(),
        )
        self.feature_fc = nn.Sequential(
            nn.Linear((25 * 25 * 8 + 64 + 4), 256),
            nn.LayerNorm(256),
            nn.Tanh(),
        )
        self.mean_fc = nn.Linear(256, 4)
        self.mean_fc.weight.data.uniform_(-init_w, init_w)
        self.mean_fc.bias.data.uniform_(-init_w, init_w)

        self.log_std_fc = nn.Linear(256, 4)
        self.log_std_fc.weight.data.uniform_(-init_w, init_w)
        self.log_std_fc.bias.data.uniform_(-init_w, init_w)

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
        mean = self.mean_fc(feature)
        log_std = self.log_std_fc(feature)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def evaluate(self, img, info, alive, epsilon=1e-6):
        mean, log_std = self.forward(img, info, alive)
        std = log_std.exp()

        normal = Normal(0, 1)
        z = normal.sample()
        z = z.cuda()
        action = torch.tanh(mean + std * z)
        log_prob = Normal(mean, std).log_prob(mean + std * z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        course = action[:, 0] * 180
        radar = (action[:, 1] + 1) * self.radar_num / 2
        disturb = (action[:, 2] + 1) * (self.radar_num + 1) / 2
        attack = (action[:, 3] + 1) * self.attack_num / 2

        course = torch.unsqueeze(course, 1)
        radar = torch.unsqueeze(radar, 1)
        disturb = torch.unsqueeze(disturb, 1)
        attack = torch.unsqueeze(attack, 1)
        act = torch.cat([course, radar, disturb, attack], 1)
        return act, log_prob, z, mean, log_std

    def get_action(self, img, info, alive):
        img = torch.unsqueeze(torch.FloatTensor(img), 0)
        info = torch.unsqueeze(torch.FloatTensor(info), 0)
        alive = torch.tensor(alive, dtype=torch.float32)
        alive = alive.view(1, 1)
        if torch.cuda.is_available():
            img = img.cuda()
            info = info.cuda()
            alive = alive.cuda()
        action, log_prob, z, mean, log_std = self.evaluate(img, info, alive)
        new_action = []
        try:
            new_action.append(int(action[0, 0]))
            new_action.append(int(action[0, 1]))
            new_action.append(int(action[0, 2]))
            new_action.append(int(action[0, 3]))
        except:
            new_action = [180, 1, 0, 0]

        return new_action


class RLFighter:
    def __init__(
            self,
            name,
            agent_num,
            attack_num,
            fighter_num,
            radar_num,
            value_lr=3e-4,
            sq_lr=3e-4,
            policy_lr=3e-4,
            reward_decay=0.99,
            tau=0.95,
            e_greedy=0.9,
            replace_target_iter=50,
            max_memory_size=5e4,
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
        self.radar_num = radar_num  # 雷达频点数
        self.batch_size = batch_size
        self.noise_rate = 5  # 噪声率（用于为动作添加随机扰动）
        self.value_lr = value_lr  # value网络学习率
        self.sq_lr = sq_lr  # soft-q网络学习率
        self.policy_lr = policy_lr  # policy网络学习率
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
        self.net_value, self.target_net_value = NetFighterValue(), NetFighterValue()
        self.net_sq = NetFighterSQ(self.agent_num)
        self.net_policy = NetFighterPolicy(attack_num=self.attack_num, radar_num=self.radar_num)

        # m_writer = SummaryWriter('runs/SAC')
        # # m_writer.add_graph(model=self.net_value, input_to_model=[torch.rand([1, 5, 100, 100]), torch.rand([1, 3])])
        # # m_writer.add_graph(model=self.net_sq, input_to_model=[torch.rand([1, 5, 100, 100]), torch.rand([1, 3]), torch.rand([1, 80])])
        # m_writer.add_graph(model=self.net_policy, input_to_model=[torch.rand([1, 5, 100, 100]), torch.rand([1, 3]), torch.rand([1,1])])
        # m_writer.close()


        # 初始化网络参数
        for m1, m2, m3 in zip(self.net_value.modules(), self.net_sq.modules(), self.net_policy.modules()):
            if isinstance(m1, nn.Conv2d) or isinstance(m1, nn.Linear):
                init.kaiming_uniform_(m1.weight, mode='fan_in', nonlinearity='leaky_relu')
            if isinstance(m2, nn.Conv2d) or isinstance(m2, nn.Linear):
                init.kaiming_uniform_(m2.weight, mode='fan_in', nonlinearity='leaky_relu')
            if isinstance(m3, nn.Conv2d) or isinstance(m3, nn.Linear):
                init.kaiming_uniform_(m3.weight, mode='fan_in', nonlinearity='leaky_relu')

        self.copy_param(self.net_value, self.target_net_value, 0)

        if self.gpu_enable:
            print('GPU Available!!')
            self.net_value = self.net_value.cuda()
            self.target_net_value = self.target_net_value.cuda()
            self.net_sq = self.net_sq.cuda()
            self.net_policy = self.net_policy.cuda()


        self.loss_func = nn.MSELoss()

        # 加载已有的模型参数
        # if self.load_state:
        #     state_dict = torch.load(self.model_path)
        #     self.eval_net_critic_fighter.load_state_dict(state_dict)
        #     self.target_net_critic_fighter.load_state_dict(state_dict)
        #     self.learn_step_counter = int(self.model_path[-10:-4])

        # 优化器
        # Adam
        self.value_optim = torch.optim.Adam(self.net_value.parameters(), lr=self.value_lr)
        self.sq_optim = torch.optim.Adam(self.net_sq.parameters(), lr=self.sq_lr)
        self.policy_optim = torch.optim.Adam(self.net_policy.parameters(), lr=self.policy_lr)

        # RMSprop
        # self.critic_optimizer_fighter = torch.optim.RMSprop(self.eval_net_critic_fighter.parameters(), lr=self.lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
        # self.actor_optimizer_fighter = torch.optim.RMSprop(self.eval_net_actor_fighter.parameters(), lr=self.lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)

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

    def learn(self, save_path, writer):
        # 复制参数+保存参数
        # learn50次复制/保存一次参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.copy_param(self.net_value, self.target_net_value, self.tau)
            print('\ntarget_params_replaced\n')
            step_counter_str = '%09d' % self.learn_step_counter
            torch.save(self.net_value.state_dict(),
                       save_path + '/value/model_' + step_counter_str + '.pkl')
            torch.save(self.net_sq.state_dict(),
                       save_path + '/soft-Q/model_' + step_counter_str + '.pkl')
            torch.save(self.net_policy.state_dict(),
                       save_path + '/policy/model_' + step_counter_str + '.pkl')
            if self.pkl_counter < self.max_pkl:
                self.pkl_counter += 1
            else:
                # 删除最旧的模型参数
                value_path = os.path.join(save_path, 'value')
                sq_path = os.path.join(save_path, 'soft-Q')
                policy_path = os.path.join(save_path, 'policy')

                files = os.listdir(value_path)
                for file in files:
                    if file.endswith('pkl'):
                        os.remove(os.path.join(value_path, file))
                        break
                files = os.listdir(sq_path)
                for file in files:
                    if file.endswith('pkl'):
                        os.remove(os.path.join(sq_path, file))
                        break
                files = os.listdir(policy_path)
                for file in files:
                    if file.endswith('pkl'):
                        os.remove(os.path.join(policy_path, file))
                        break

        # 采样Replay
        [s_screen_batch, s_info_batch, alive_batch, alive__batch, self_a_batch,
         mate_a_batch, other_a_batch, r_batch, s__screen_batch, s__info_batch] = self.memory.sample_replay(
            self.batch_size, self.gpu_enable)

        self.value_optim.zero_grad()
        self.sq_optim.zero_grad()
        self.policy_optim.zero_grad()

        self_a_batch = torch.unsqueeze(self_a_batch, 1)
        memory_action = torch.cat([self_a_batch, mate_a_batch, other_a_batch], 1).view(mate_a_batch.size(0), -1)
        q_value = self.net_sq(s_screen_batch, s_info_batch, memory_action)
        value = self.net_value(s_screen_batch, s_info_batch)
        new_action, log_prob, z, mean, log_std = self.net_policy.evaluate(s_screen_batch, s_info_batch, alive_batch)
        new_action = torch.unsqueeze(new_action, 1)
        new_all_action = torch.cat([new_action, mate_a_batch, other_a_batch], 1).view(mate_a_batch.size(0), -1)

        target_value = self.target_net_value(s__screen_batch, s__info_batch)
        q_next = (r_batch + self.gamma * target_value).detach()
        q_value_loss = self.loss_func(q_value, q_next)
        new_q_value = self.net_sq(s_screen_batch, s_info_batch, new_all_action)
        next_value = (new_q_value - log_prob).detach()
        value_loss = self.loss_func(value, next_value)

        # policy_loss有争议
        # policy_loss = (log_prob - new_q_value).mean()

        target_log_prob = new_q_value - value
        policy_loss = (log_prob * (log_prob - target_log_prob).detach()).mean()

        value_loss.backward()
        nn.utils.clip_grad_norm_(self.net_value.parameters(), max_norm=5, norm_type=2)
        self.value_optim.step()

        q_value_loss.backward()
        nn.utils.clip_grad_norm_(self.net_sq.parameters(), max_norm=5, norm_type=2)
        self.sq_optim.step()

        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.net_policy.parameters(), max_norm=5, norm_type=2)
        self.policy_optim.step()

        print("learn_step: %d, %s_value_loss: %.4f, %s_q_loss: %.4f, %s_policy_loss: %.4f, mean_reward: %.2f" % (
            self.learn_step_counter, self.name, value_loss, self.name, q_value_loss, self.name, policy_loss, torch.mean(r_batch)))

        # 训练过程保存
        writer.add_scalar(tag='%s_value_loss' % self.name, scalar_value=value_loss, global_step=self.learn_step_counter)
        writer.add_scalar(tag='%s_q_loss' % self.name, scalar_value=q_value_loss, global_step=self.learn_step_counter)
        writer.add_scalar(tag='%s_policy_loss' % self.name, scalar_value=policy_loss, global_step=self.learn_step_counter)

        self.learn_step_counter += 1