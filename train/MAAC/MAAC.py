import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import os
from common.Replay2 import Memory
from train.MAAC.Critic import NetFighterCritic
from train.MAAC.Actor import NetFighterActor


def copy_param(eval_net, target_net, tau):
    # 将eval网络中的参数复制到target网络中
    # tau: target网络参数保留率
    tem_dict = {}
    for param_tensor in eval_net.state_dict():
        tem_value = tau * target_net.state_dict()[param_tensor] + (1 - tau) * eval_net.state_dict()[param_tensor]
        tem_dict[param_tensor] = tem_value
    target_net.load_state_dict(tem_dict)


class RLFighter:
    def __init__(
            self,
            name,
            agent_num,
            attack_num,
            fighter_num,
            radar_num,
            actor_lr=0.01,
            critic_lr=0.01,
            reward_decay=0.95,
            tau=0.99,
            e_greedy=0.9,
            replace_target_iter=50,
            max_memory_size=1e5,
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
        self.eval_net_actor_fighters = [NetFighterActor() for i in range(self.agent_num)]
        self.target_net_actor_fighters = [NetFighterActor() for i in range(self.agent_num)]

        # 初始化网络参数
        for actor in self.eval_net_actor_fighters:
            for m1 in actor.modules():
                if isinstance(m1, nn.Conv2d) or isinstance(m1, nn.Linear):
                    init.kaiming_uniform_(m1.weight, mode='fan_in', nonlinearity='leaky_relu')

        copy_param(self.eval_net_critic_fighter, self.target_net_critic_fighter, 0)
        for i in range(self.agent_num):
            copy_param(self.eval_net_actor_fighters[i], self.target_net_actor_fighters[i], 0)

        if self.gpu_enable:
            print('GPU Available!!')
            self.eval_net_critic_fighter = self.eval_net_critic_fighter.cuda()
            self.target_net_critic_fighter = self.target_net_critic_fighter.cuda()
            for i in range(self.agent_num):
                self.eval_net_actor_fighters[i] = self.eval_net_actor_fighters[i].cuda()
                self.target_net_actor_fighters[i] = self.target_net_actor_fighters[i].cuda()

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
        self.actor_optimizer_fighters = [torch.optim.Adam(self.eval_net_actor_fighters[i].parameters(), lr=self.ac_lr) for i in range(self.agent_num)]

    def store_replay(self, s_screen, s_info, alive, alive_, a, r, s__screen, s__info):
        # 将每一step的经验加入经验池
        self.memory.store_replay(s_screen, s_info, alive, alive_, a, r, s__screen, s__info)

    def get_memory_size(self):
        return self.memory.get_size()

    # 选择行为(只针对单个数据，不是batch，用于决策时)
    def choose_action(self, img_obs, info_obs, agent_id):
        img_obs = torch.unsqueeze(torch.FloatTensor(img_obs), 0)
        info_obs = torch.unsqueeze(torch.FloatTensor(info_obs), 0)
        if self.gpu_enable:
            img_obs = img_obs.cuda()
            info_obs = info_obs.cuda()

        act = self.target_net_actor_fighters[agent_id](img_obs, info_obs).cpu()
        course = torch.tanh(act[:, 0]) * 180
        try:
            # 航向
            course = int(course)
            course = int(course + self.noise_rate * np.random.randn())  # 对航向添加一个随机扰动

            # 雷达频点
            radar = act[:, 1]
            radar = int(1 / (1 + torch.exp(-radar)) * self.radar_num)

            # 干扰频点
            disturb = act[:, 2]
            disturb = int(1 / (1 + torch.exp(-disturb)) * (self.radar_num + 1))

            # 攻击
            attack = act[:, 3]  # 攻击
            attack = int(1 / (1 + torch.exp(-attack)) * self.attack_num)

        except:
            course = 179
            radar = 1
            disturb = 0
            attack = 0
        return [course, radar, disturb, attack]

    # 针对batch的数据选择行为(使用eval_actor网络计算，用于训练)
    def choose_action_batch(self, img_obs_batch, info_obs_batch):
        if self.gpu_enable:
            img_obs_batch = img_obs_batch.cuda()
            info_obs_batch = info_obs_batch.cuda()

        acts = []
        for i in range(self.agent_num):
            acts.append(self.eval_net_actor_fighters[i](img_obs_batch[i], info_obs_batch[i]))

        actions = []
        for act in acts:
            # 航向
            course = torch.tanh(act[:, 0]) * 180

            # 雷达
            radar = 1 / (1 + torch.exp(-act[:, 1])) * self.radar_num

            # 干扰
            disturb = 1 / (1 + torch.exp(-act[:, 2])) * (self.radar_num + 1)

            # 攻击
            attack = act[:, 3]
            attack = 1 / (1 + torch.exp(-attack)) * self.attack_num

            course = torch.unsqueeze(course, 1)
            radar = torch.unsqueeze(radar, 1)
            disturb = torch.unsqueeze(disturb, 1)
            attack = torch.unsqueeze(attack, 1)
            action = torch.cat([course, radar, disturb, attack], 1)
            actions.append(action)
        return actions

    # 针对batch的数据选择行为(使用target_actor网络计算)
    def choose_action_batch_tar(self, img_obs_batch, info_obs_batch):
        noise = torch.randn(self.batch_size, 1) * self.noise_rate
        if self.gpu_enable:
            img_obs_batch = img_obs_batch.cuda()
            info_obs_batch = info_obs_batch.cuda()
            noise = noise.cuda()

        acts = []
        for i in range(self.agent_num):
            acts.append(self.target_net_actor_fighters[i](img_obs_batch[i], info_obs_batch[i]))

        actions = []
        for act in acts:
            # 航向
            course = torch.tanh(act[:, 0]) * 180

            # 雷达
            radar = 1 / (1 + torch.exp(-act[:, 1])) * self.radar_num

            # 干扰
            disturb = 1 / (1 + torch.exp(-act[:, 2])) * (self.radar_num + 1)

            # 攻击
            attack = act[:, 3]
            attack = 1 / (1 + torch.exp(-attack)) * self.attack_num

            course = torch.unsqueeze(course, 1)
            radar = torch.unsqueeze(radar, 1)
            disturb = torch.unsqueeze(disturb, 1)
            attack = torch.unsqueeze(attack, 1)
            action = torch.cat([course, radar, disturb, attack], 1)
            actions.append(action)
        return actions

    def update_critic(self, r_batches, alive_batches, q_next, q_eval, regs):
        self.critic_optimizer_fighter.zero_grad()
        critic_loss = 0
        for i in range(self.agent_num):
            q_target = r_batches[i] + self.gamma * q_next[i] * alive_batches[i]
            critic_loss = critic_loss + self.loss_func(q_eval[i], q_target.detach())
            critic_loss = critic_loss + regs[i]  # 正则化项

        critic_loss.backward()
        self.eval_net_critic_fighter.scale_shared_grads()
        # nn.utils.clip_grad_norm_(self.eval_net_critic_fighter.parameters(), max_norm=5, norm_type=2)
        nn.utils.clip_grad_norm_(self.eval_net_critic_fighter.parameters(), 100)
        self.critic_optimizer_fighter.step()
        return critic_loss

    def update_actor(self, s_screen_batches, s_info_batches):
        actions = []
        # 禁用critic网络梯度
        for p in self.eval_net_critic_fighter.parameters():
            p.required_grad = False

        for i in range(self.agent_num):
            screen = s_screen_batches[i]
            info = s_info_batches[i]
            act = self.eval_net_actor_fighters[i](screen, info)
            actions.append(act)
        actor_losses = []
        for i in range(self.agent_num):
            self.actor_optimizer_fighters[i].zero_grad()
            q, _ = self.eval_net_critic_fighter(s_screen_batches, s_info_batches, actions)
            qi = q[i]
            act_loss = -1.0 * qi.mean()
            # act_reg = torch.mean(torch.square(actions[i]))
            actor_loss = act_loss  # + act_reg * 1e-3
            actor_losses.append(actor_loss)
            actor_losses[i].backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.eval_net_actor_fighters[i].parameters(), max_norm=5, norm_type=2)
            self.actor_optimizer_fighters[i].step()
        # 启用critic网络梯度
        for p in self.eval_net_critic_fighter.parameters():
            p.required_grad = True
        return actor_losses

    def learn(self, save_path, batch_indexes, writer, red_mem):
        # 复制参数+保存参数
        # learn50次复制/保存一次参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            copy_param(self.eval_net_critic_fighter, self.target_net_critic_fighter, self.tau)
            for i in range(self.agent_num):
                copy_param(self.eval_net_actor_fighters[i], self.target_net_actor_fighters[i], self.tau)
            print('\ntarget_params_replaced\n')
            # step_counter_str = '%09d' % self.learn_step_counter
            # torch.save(self.target_net_critic_fighter.state_dict(),
            #            save_path + '/critic/model_' + step_counter_str + '.pkl')
            # torch.save(self.target_net_actor_fighter.state_dict(),
            #            save_path + '/actor/model_' + step_counter_str + '.pkl')
            # if self.pkl_counter < self.max_pkl:
            #     self.pkl_counter += 1
            # else:
            #     # 删除最旧的模型参数
            #     critic_path = os.path.join(save_path, 'critic')
            #     actor_path = os.path.join(save_path, 'actor')
            #
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
        [s_screen_batches, s_info_batches, alive_batches, alive__batches, self_a_batches,
         r_batches, s__screen_batches, s__info_batches] = self.memory.sample_replay(batch_indexes)
        self_a_batch = self_a_batches.view(-1, 4 * self.agent_num, 1)

        s_screen_batches = s_screen_batches.permute(1, 0, 2, 3, 4)
        s_info_batches = s_info_batches.permute(1, 0, 2)
        r_batches = r_batches.permute(1, 0, 2)
        alive_batches = alive_batches.permute(1, 0, 2)
        s__screen_batches = s__screen_batches.permute(1, 0, 2, 3, 4)
        s__info_batches = s__info_batches.permute(1, 0, 2)

        # 反向传播、优化
        # critic
        # 计算得到所有agent现在时刻与下一时刻的action
        # actions = self.choose_action_batch(s_screen_batches, s_info_batches)
        actions = self_a_batches.permute(1, 0, 2)
        actions_ = self.choose_action_batch_tar(s__screen_batches, s__info_batches)

        q_eval, regs = self.eval_net_critic_fighter(s_screen_batches, s_info_batches, actions)
        q_next, _ = self.target_net_critic_fighter(s__screen_batches, s__info_batches, actions_)
        critic_loss = self.update_critic(r_batches, alive_batches, q_next, q_eval, regs)

        # actor
        actor_losses = self.update_actor(s_screen_batches, s_info_batches)

        self.cost_critic_his.append(critic_loss)
        self.cost_actor_his.append(actor_losses)
        print("learn_step: %d, %s_critic_loss: %.4f, %s_actor_loss: %.4f, mean_reward: %.2f" % (
            self.learn_step_counter, self.name, critic_loss, self.name, actor_losses.mean(), torch.mean(r_batches)))

        # 训练过程保存
        writer.add_scalar(tag='%s_actor_loss' % self.name, scalar_value=actor_losses, global_step=self.learn_step_counter)
        writer.add_scalar(tag='%s_critic_loss' % self.name, scalar_value=critic_loss, global_step=self.learn_step_counter)

        self.learn_step_counter += 1
