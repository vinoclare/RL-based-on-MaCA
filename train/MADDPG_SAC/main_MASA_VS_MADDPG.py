"""
一方为MADDPG,一方为fix-rule-no-attack的训练
"""
import sys
import os

root_path = 'C:/MaCA'
env_path = os.path.join(root_path, 'environment')

sys.path.append(root_path)
sys.path.append(env_path)

import copy
import math
import torch
import torch.nn as nn
import numpy as np
from agent.fix_rule_no_att.agent import Agent
from interface import Environment
import random
from train.MADDPG_SAC import MADDPG_SAC_ATTENTION as MASA
from train.MADDPG_SAC import MADDPG_4 as MADDPG
from torch.utils.tensorboard import SummaryWriter
from configuration.reward import GlobalVar as REWARD
from Attention import Attention

MAP_PATH = os.path.join(root_path, 'maps/1000_1000_fighter10v10.map')

RENDER = True  # 是否渲染，渲染能加载出实时的训练画面，但是会降低训练速度
MAX_EPOCH = 2000
BATCH_SIZE = 256
GAMMA = 0.99  # reward discount
TAU = 0.99
BETA = 0  # 边界惩罚discount
ALPHA = 0.2  # 温度系数
replace_target_iter = 10  # target网络更新频率
MAX_STEP = 1999  # 1个epoch内最大步数
LEARN_INTERVAL = 500  # 学习间隔
pass_step = 10  # 间隔x个step保存一次经验
reward_scaling = 0.3  # reward缩放系数

# 网络学习率
actor_lr = 1e-5
critic_lr = 1e-5

DETECTOR_NUM = 0
FIGHTER_NUM = 10
MAX_MEM_SIZE = 1e4  # 经验回放池最大容量
ATTACK_IND_NUM = (DETECTOR_NUM + FIGHTER_NUM) * 2 + 1  # 导弹攻击类型数量（不攻击+中程导弹攻击目标+远程导弹攻击目标）
RADAR_NUM = 10  # 雷达频点总数

# COURSE_NUM = 16
# ACTION_NUM = COURSE_NUM * ATTACK_IND_NUM

# 清除tensorboard文件
runs_path = os.path.join(root_path, 'runs/Battle')
if not os.path.exists(runs_path):
    os.makedirs(runs_path)
for file in os.listdir(runs_path):
    path = os.path.join(runs_path, file)
    os.remove(path)


def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_critic(agents, Attention, critic_optimizer_fighters, batch_indexes, s_screen_batches, s_info_batches, mate_a_batches,
                 s__screen_batches, s__info_batches, r_batches, alive_batches, done_batches, self_a_batches, other_a_batches):
    with torch.autograd.set_detect_anomaly(True):
        e1, e2 = [], []
        e1_, e2_ = [], []
        loss_func = nn.MSELoss()
        critic_loss_mean = 0
        for i in range(len(other_a_batches)):
            if i == 0:
                other_a_batch = other_a_batches[i]
            else:
                other_a_batch = torch.cat((other_a_batch, other_a_batches[i]), 1)

        log_probs_es = []
        for i in range(len(agents)):
            action_, log_probs_ = agents[i].choose_action_batch(s__screen_batches[i], s__info_batches[i])
            action_ = torch.unsqueeze(action_, 1)
            log_probs_es.append(log_probs_)
            all_action_ = torch.cat([action_, mate_a_batches[i], other_a_batch], 1).view(mate_a_batches[i].size(0), -1)
            all_mem_action = torch.cat([self_a_batches[i], mate_a_batches[i], other_a_batch], 1).view(mate_a_batches[i].size(0), -1)
            e1_next, e2_next = agents[i].target_net_critic_fighter.encoding(s__screen_batches[i], s__info_batches[i], all_action_)
            e1_cur, e2_cur = agents[i].eval_net_critic_fighter.encoding(s_screen_batches[i], s_info_batches[i], all_mem_action)
            e1.append(e1_next)
            e2.append(e2_next)
            e1_.append(e1_cur)
            e2_.append(e2_cur)

        attentions1 = Attention(e1)
        attentions2 = Attention(e2)
        attentions1_ = Attention(e1_)
        attentions2_ = Attention(e2_)

        for i in range(len(agents)):
            q1_next, q2_next = agents[i].target_net_critic_fighter.decoding(attentions1[i], attentions2[i])
            q_next = (torch.min(q1_next, q2_next) - ALPHA * log_probs_es[i]).detach()
            q_target = reward_scaling * r_batches[i] + alive_batches[i] * GAMMA * q_next * (1 - done_batches[i])

            q1_cur, q2_cur = agents[i].eval_net_critic_fighter.decoding(attentions1_[i], attentions2_[i])

            critic_loss = loss_func(q1_cur, q_target) + loss_func(q2_cur, q_target)
            critic_loss_mean += critic_loss
            critic_optimizer_fighters[i].zero_grad()
            critic_loss.backward(retain_graph=True)

        # 梯度缩放
        Attention.scale_grads()

        for i in range(len(agents)):
            critic_optimizer_fighters[i].step()
    return critic_loss_mean/10


def train_actor(agents, Attention, policy_optimizer_fighters, batch_indexes, s_screen_batches, s_info_batches, mate_a_batches,
                s__screen_batches, s__info_batches, r_batches, alive_batches, done_batches, self_a_batches, other_a_batches):
    with torch.autograd.set_detect_anomaly(True):
        e1 = []
        e2 = []
        actor_loss_mean = 0

        for i in range(len(other_a_batches)):
            if i == 0:
                other_a_batch = other_a_batches[i]
            else:
                other_a_batch = torch.cat((other_a_batch, other_a_batches[i]), 1)

        for i in range(len(agents)):
            action1, log_probs = agents[i].choose_action_batch(s_screen_batches[i], s_info_batches[i])
            action1 = torch.unsqueeze(action1, 1)
            all_action = torch.cat([action1, mate_a_batches[i], other_a_batch], 1).view(mate_a_batches[i].size(0), -1)
            e1_pi, e2_pi = agents[i].eval_net_critic_fighter.encoding(s_screen_batches[i], s_info_batches[i], all_action)
            e1.append(e1_pi)
            e2.append(e2_pi)

        attentions1 = Attention(e1)
        attentions2 = Attention(e2)

        for i in range(len(agents)):
            q1_pi, q2_pi = agents[i].eval_net_critic_fighter.decoding(attentions1[i], attentions2[i])
            min_q_pi = torch.min(q1_pi, q2_pi)
            action2, log_probs2 = agents[i].choose_action_batch(s_screen_batches[i], s_info_batches[i])
            actor_loss = ((ALPHA * log_probs2) - min_q_pi).mean()
            actor_loss_mean += actor_loss

            policy_optimizer_fighters[i].zero_grad()
            actor_loss.backward(retain_graph=True)
            # nn.utils.clip_grad_norm_(self.eval_net_critic_fighter.parameters(), 0.5)
        for i in range(len(agents)):
            policy_optimizer_fighters[i].step()
    return actor_loss_mean/10


def set_value_in_img(img, pos_x, pos_y, value):
    # 向图像指定位置中插入值
    img_obs_size_x = 100
    img_obs_size_y = 100
    # 左上角
    if pos_x == 0 and pos_y == 0:
        img[pos_x: pos_x + 2, pos_y: pos_y + 2] = value
    # 左下角
    elif pos_x == 0 and pos_y == (img_obs_size_y - 1):
        img[pos_x: pos_x + 2, pos_y - 1: pos_y + 1] = value
    # 右上角
    elif pos_x == (img_obs_size_x - 1) and pos_y == 0:
        img[pos_x - 1: pos_x + 1, pos_y: pos_y + 2] = value
    # 右下角
    elif pos_x == (img_obs_size_x - 1) and pos_y == (img_obs_size_y - 1):
        img[pos_x - 1: pos_x + 1, pos_y - 1: pos_y + 1] = value
    # 左边
    elif pos_x == 0:
        img[pos_x: pos_x + 2, pos_y - 1: pos_y + 2] = value
    # 右边
    elif pos_x == img_obs_size_x - 1:
        img[pos_x - 1: pos_x + 1, pos_y - 1: pos_y + 2] = value
    # 上边
    elif pos_y == 0:
        img[pos_x - 1: pos_x + 2, pos_y: pos_y + 2] = value
    # 下边
    elif pos_y == img_obs_size_y - 1:
        img[pos_x - 1: pos_x + 2, pos_y - 1: pos_y + 1] = value
    # 其他位置
    else:
        img[pos_x - 1: pos_x + 2, pos_y - 1: pos_y + 2] = value


if __name__ == "__main__":
    seed_everything(1004)

    # 双方的obs构建模块
    red_agent_obs_ind = 'MADDPG_SAC'
    blue_agent_obs_ind = 'MADDPG_SAC'

    # 创建环境
    os.chdir(root_path)
    env = Environment(MAP_PATH, red_agent_obs_ind, blue_agent_obs_ind, render=RENDER)
    # 获取环境信息
    size_x, size_y = env.get_map_size()
    red_detector_num, red_fighter_num, blue_detector_num, blue_fighter_num = env.get_unit_num()
    # 为双方设置环境信息
    blue_detector_action = []
    blue_fighter_models = []
    red_detector_action = []
    red_fighter_models = []

    # Attention模块
    blue_attention = Attention(10).cuda()

    # 优化器
    blue_critic_optimizer_fighters = []
    blue_policy_optimizer_fighters = []
    for y in range(blue_fighter_num):
        blue_fighter_model = MASA.RLFighter(name='blue_%d' % y, agent_num=(DETECTOR_NUM + FIGHTER_NUM) * 2,
                                            attack_num=ATTACK_IND_NUM, fighter_num=FIGHTER_NUM, radar_num=RADAR_NUM,
                                            max_memory_size=MAX_MEM_SIZE, replace_target_iter=replace_target_iter,
                                            actor_lr=actor_lr, critic_lr=critic_lr, reward_decay=GAMMA,
                                            tau=TAU, batch_size=BATCH_SIZE)
        blue_fighter_models.append(blue_fighter_model)
        blue_critic_optimizer_fighters.append(torch.optim.Adam(blue_fighter_model.eval_net_critic_fighter.parameters(), lr=critic_lr))
        blue_policy_optimizer_fighters.append(torch.optim.Adam(blue_fighter_model.policy_net_fighter.parameters(), lr=actor_lr))

    for y in range(red_fighter_num):
        red_fighter_model = MADDPG.RLFighter(name='red_%d' % y, agent_num=(DETECTOR_NUM + FIGHTER_NUM) * 2,
                                             attack_num=ATTACK_IND_NUM, fighter_num=FIGHTER_NUM, radar_num=RADAR_NUM,
                                             max_memory_size=MAX_MEM_SIZE, replace_target_iter=replace_target_iter,
                                             actor_lr=actor_lr, critic_lr=critic_lr, reward_decay=GAMMA,
                                             tau=TAU, batch_size=BATCH_SIZE)
        red_fighter_models.append(red_fighter_model)

    # 加载预存储的数据
    for i in range(len(blue_fighter_models)):
        path = os.path.join('prerun_data', 'MASA_VS_MADDPG', 'blue', '%d_data.npy' % i)
        blue_fighter_models[i].load_from_file(path)

    for i in range(len(red_fighter_models)):
        path = os.path.join('prerun_data', 'MASA_VS_MADDPG', 'red', '%d_data.npy' % i)
        red_fighter_models[i].load_from_file(path)

    print('PreData loaded!')

    writer = SummaryWriter('runs/Battle')

    for y in range(blue_fighter_num):
        if not os.path.exists('model/Battle/blue/%d/critic' % y):
            os.makedirs('model/Battle/blue/%d/critic' % y)
        if not os.path.exists('model/Battle/blue/%d/actor' % y):
            os.makedirs('model/Battle/blue/%d/actor' % y)
        if not os.path.exists('model/Battle/red/%d/critic' % y):
            os.makedirs('model/Battle/red/%d/critic' % y)
        if not os.path.exists('model/Battle/red/%d/actor' % y):
            os.makedirs('model/Battle/red/%d/actor' % y)
    if not os.path.exists('model/Battle/blue/Attention'):
        os.makedirs('model/Battle/blue/Attention')

    # 训练循环
    global_step_cnt = 0
    learn_step_counter = 0
    blue_epoch_rewards = []
    red_epoch_rewards = []
    wines = []
    for x in range(MAX_EPOCH):
        print("Epoch: %d" % x)
        step_cnt = 0
        env.reset()  # 重置环境
        blue_epoch_reward = 0  # 记录一个epoch内的蓝方平均reward
        red_epoch_reward = 0
        while True:
            # 获取双方初始环境观测
            if step_cnt == 0:
                red_obs_list = []
                red_obs_dict, blue_obs_dict = env.get_obs()

            # 双方的全部Agent位置
            # 红色方agent位置
            red_poses = []
            for i in range(FIGHTER_NUM):
                if red_obs_dict['fighter'][i]['alive']:
                    tem_dict = {'id': i + 1, 'pos_x': red_obs_dict['fighter'][i]['pos_x'],
                                'pos_y': red_obs_dict['fighter'][i]['pos_y']}
                    red_poses.append(tem_dict)

            # 蓝色方agent位置
            blue_poses = []
            for i in range(FIGHTER_NUM):
                if blue_obs_dict['fighter'][i]['alive']:
                    tem_dict = {'id': i + 1, 'pos_x': blue_obs_dict['fighter'][i]['pos_x'],
                                'pos_y': blue_obs_dict['fighter'][i]['pos_y']}
                    blue_poses.append(tem_dict)

            # 获取蓝色方行动
            blue_alive = []  # 蓝队全队存活信息
            blue_obs_list = []  # 蓝色方的全体环境观测信息
            blue_actions = []
            blue_fighter_action = []  # 蓝色方所有agent的行动
            for y in range(blue_fighter_num):
                if blue_obs_dict['fighter'][y]['alive']:  # 存活
                    tmp_img_obs = blue_obs_dict['fighter'][y]['screen']
                    tmp_img_obs = tmp_img_obs.transpose(2, 0, 1)

                    # 向图中添加全局信息
                    for dic in red_poses:
                        set_value_in_img(tmp_img_obs[1], int(dic['pos_y']/10), int(dic['pos_x']/10), 90 + dic['id'] * 10)

                    tmp_info_obs = blue_obs_dict['fighter'][y]['info']
                    alive = 1
                    blue_alive.append(alive)
                    true_action, action = blue_fighter_models[y].choose_action(tmp_img_obs, tmp_info_obs)
                    blue_obs_list.append({'screen': copy.deepcopy(tmp_img_obs), 'info': copy.deepcopy(tmp_info_obs)})
                    blue_fighter_action.append(true_action)
                    blue_actions.append(action)
                else:  # 阵亡
                    tmp_img_obs = blue_obs_dict['fighter'][y]['screen']
                    tmp_img_obs = tmp_img_obs.transpose(2, 0, 1)

                    # 向图中添加全局信息
                    for dic in red_poses:
                        set_value_in_img(tmp_img_obs[1], int(dic['pos_y'] / 10), int(dic['pos_x'] / 10),
                                         90 + dic['id'] * 10)

                    tmp_info_obs = blue_obs_dict['fighter'][y]['info']
                    alive = 0
                    blue_alive.append(alive)
                    true_action = np.array([0, 0, 0, 0], dtype=np.int32)
                    action = np.array([0, 0, 0, 0], dtype=np.int32)
                    blue_obs_list.append({'screen': copy.deepcopy(tmp_img_obs), 'info': copy.deepcopy(tmp_info_obs)})
                    blue_fighter_action.append(true_action)
                    blue_actions.append(action)

            blue_actions = np.array(blue_actions)
            blue_fighter_action = np.array(blue_fighter_action)

            # 获取红色方行动
            red_alive = []
            red_obs_list = []
            red_actions = []
            red_fighter_action = []
            for y in range(red_fighter_num):
                if red_obs_dict['fighter'][y]['alive']:
                    tmp_img_obs = red_obs_dict['fighter'][y]['screen']
                    tmp_img_obs = tmp_img_obs.transpose(2, 0, 1)

                    # 向图中添加全局信息
                    for dic in blue_poses:
                        set_value_in_img(tmp_img_obs[1], int(dic['pos_y']/10), int(dic['pos_x']/10), 90 + dic['id'] * 10)

                    tmp_info_obs = red_obs_dict['fighter'][y]['info']
                    alive = 1
                    red_alive.append(alive)
                    true_action, action = red_fighter_models[y].choose_action(tmp_img_obs, tmp_info_obs)
                    true_action[0] = true_action[0] + 180
                    red_obs_list.append({'screen': copy.deepcopy(tmp_img_obs), 'info': copy.deepcopy(tmp_info_obs)})
                    red_fighter_action.append(true_action)
                    red_actions.append(action)
                else:
                    tmp_img_obs = red_obs_dict['fighter'][y]['screen']
                    tmp_img_obs = tmp_img_obs.transpose(2, 0, 1)

                    # 向图中添加全局信息
                    for dic in blue_poses:
                        set_value_in_img(tmp_img_obs[1], int(dic['pos_y'] / 10), int(dic['pos_x'] / 10),
                                         90 + dic['id'] * 10)

                    tmp_info_obs = red_obs_dict['fighter'][y]['info']
                    alive = 0
                    red_alive.append(alive)
                    true_action, action = np.array([0, 0, 0, 0], dtype=np.int32), np.array([0, 0, 0, 0], dtype=np.int32)
                    red_obs_list.append({'screen': copy.deepcopy(tmp_img_obs), 'info': copy.deepcopy(tmp_info_obs)})
                    red_fighter_action.append(true_action)
                    red_actions.append(action)

            red_actions = np.array(red_actions)
            red_fighter_action = np.array(red_fighter_action)

            # step X 1
            env.step(red_detector_action, red_fighter_action, blue_detector_action, blue_fighter_action)
            step_cnt += 1
            # 获取reward
            red_detector_reward, red_fighter_reward, red_game_reward, blue_detector_reward, \
                blue_fighter_reward, blue_game_reward = env.get_reward()
            blue_step_reward = blue_fighter_reward + blue_game_reward
            red_step_reward = red_fighter_reward + red_game_reward

            # step X pass_step-1
            for i in range(pass_step-1):
                # 获取红色方行动
                env.step(red_detector_action, red_fighter_action, blue_detector_action, blue_fighter_action)

                # 获取reward
                red_detector_reward, red_fighter_reward, red_game_reward, blue_detector_reward, \
                    blue_fighter_reward, blue_game_reward = env.get_reward()
                blue_step_reward += blue_fighter_reward + blue_game_reward
                red_step_reward += red_fighter_reward + red_game_reward
                step_cnt += 1

            # 双方的全部Agent位置
            # 红色方agent位置
            red_poses = []
            red_alives = 0
            for i in range(FIGHTER_NUM):
                if red_obs_dict['fighter'][i]['alive']:
                    tem_dict = {'id': i + 1, 'pos_x': red_obs_dict['fighter'][i]['pos_x'],
                                'pos_y': red_obs_dict['fighter'][i]['pos_y']}
                    red_poses.append(tem_dict)
                    red_alives += 1

            # 蓝色方agent位置
            blue_poses = []
            blue_alives = 0
            for i in range(FIGHTER_NUM):
                if blue_obs_dict['fighter'][i]['alive']:
                    tem_dict = {'id': i + 1, 'pos_x': blue_obs_dict['fighter'][i]['pos_x'],
                                'pos_y': blue_obs_dict['fighter'][i]['pos_y']}
                    blue_poses.append(tem_dict)
                    blue_alives += 1

            # 判断胜负
            if env.get_done() or step_cnt > MAX_STEP:
                num_diff = blue_alives - red_alives
                if num_diff > 7:
                    win = 2
                elif num_diff > 3:
                    win = 1
                elif num_diff < -7:
                    win = -2
                elif num_diff < -3:
                    win = -1
                elif num_diff > 0:
                    win = 0.5
                elif num_diff < 0:
                    win = -0.5
                else:
                    win = 0
                wines.append(win)
                print('%s win!' % ('blue' if win > 0 else 'red'))
                blue_win_rate = np.sum(np.array(wines) > 0) / len(wines)
                red_win_rate = np.sum(np.array(wines) < 0) / len(wines)

            # 保存蓝色方replay
            red_obs_dict, blue_obs_dict = env.get_obs()
            for y in range(blue_fighter_num):
                tmp_img_obs = blue_obs_dict['fighter'][y]['screen']
                tmp_img_obs = tmp_img_obs.transpose(2, 0, 1)

                # 向图中添加全局信息
                for dic in red_poses:
                    set_value_in_img(tmp_img_obs[1], int(dic['pos_y']/10), int(dic['pos_x']/10), 90 + dic['id'] * 10)

                tmp_info_obs = blue_obs_dict['fighter'][y]['info']
                blue_obs_list_ = {'screen': copy.deepcopy(tmp_img_obs), 'info': copy.deepcopy(tmp_info_obs)}
                self_action = blue_fighter_action[y]
                done = 0
                if env.get_done() or step_cnt > MAX_STEP:
                    done = 1
                    if win == 2:
                        blue_step_reward[y] += REWARD.reward_totally_win/10
                    elif win == 1:
                        blue_step_reward[y] += REWARD.reward_win/10
                    if win == -2:
                        blue_step_reward[y] += REWARD.reward_totally_lose/10
                    elif win == -1:
                        blue_step_reward[y] += REWARD.reward_lose/10
                    blue_step_reward[y] += 5 * num_diff
                blue_fighter_models[y].store_replay(blue_obs_list[y], blue_alive[y], self_action,
                                                    blue_step_reward[y]/pass_step, blue_obs_list_, done)

            # 保存红色方replay
            for y in range(red_fighter_num):
                tmp_img_obs = red_obs_dict['fighter'][y]['screen']
                tmp_img_obs = tmp_img_obs.transpose(2, 0, 1)

                # 向图中添加全局信息
                for dic in blue_poses:
                    set_value_in_img(tmp_img_obs[1], int(dic['pos_y']/10), int(dic['pos_x']/10), 90 + dic['id'] * 10)

                tmp_info_obs = red_obs_dict['fighter'][y]['info']
                red_obs_list_ = {'screen': copy.deepcopy(tmp_img_obs), 'info': copy.deepcopy(tmp_info_obs)}
                self_action = red_fighter_action[y]
                done = 0
                if env.get_done() or step_cnt > MAX_STEP:
                    done = 1
                    if win == -2:
                        red_step_reward[y] += REWARD.reward_totally_win/10
                    elif win == -1:
                        red_step_reward[y] += REWARD.reward_win/10
                    if win == 2:
                        red_step_reward[y] += REWARD.reward_totally_lose/10
                    elif win == 1:
                        red_step_reward[y] += REWARD.reward_lose/10
                    red_step_reward[y] -= 5 * num_diff
                red_fighter_models[y].store_replay(red_obs_list[y], red_alive[y], self_action,
                                                   red_step_reward[y]/pass_step, red_obs_list_, done)

            global_step_cnt += 1
            blue_epoch_reward += blue_step_reward.mean()
            red_epoch_reward += red_step_reward.mean()

            # 环境判定完成后（回合完毕），开始学习模型参数
            if env.get_done():
                writer.add_scalar(tag='blue_epoch_reward', scalar_value=blue_epoch_reward,
                                      global_step=x)
                writer.add_scalar(tag='red_epoch_reward', scalar_value=red_epoch_reward,
                                      global_step=x)
                writer.add_scalar(tag='win', scalar_value=win,
                                      global_step=x)
                writer.add_scalar(tag='blue_win_rate', scalar_value=blue_win_rate,
                                      global_step=x)
                writer.add_scalar(tag='red_win_rate', scalar_value=red_win_rate,
                                      global_step=x)
                # 保存模型
                if len(blue_epoch_rewards) > 10:
                    # Blue
                    epoch_rewards2 = np.sort(blue_epoch_rewards)
                    threshold = epoch_rewards2[-5]
                    if blue_epoch_reward > threshold:
                        # 保存最优模型
                        for i in range(len(blue_fighter_models)):
                            save_path = 'model/Battle/blue/%d' % i
                            blue_fighter_models[i].save_best_model(save_path, blue_epoch_reward)
                        torch.save(blue_attention.state_dict(), 'model/Battle/blue/Attention/attention_%s.pkl' % str(int(blue_epoch_reward)))
                        att_pkls = os.listdir('model/Battle/blue/Attention/')
                        if len(att_pkls) > 5:
                            rewards = []
                            for file in att_pkls:
                                reward = int(float(file[10:-4]))
                                rewards.append(reward)
                            index = np.argmin(rewards)
                            os.remove(os.path.join('model/Battle/blue/Attention/', att_pkls[index]))

                print("blue_epoch_reward: %.3f   red_epoch_reward: %.3f" % (blue_epoch_reward, red_epoch_reward))
                blue_epoch_rewards.append(blue_epoch_reward)
                red_epoch_rewards.append(red_epoch_reward)
                break

            # 学习模型参数
            if step_cnt != 0 and (step_cnt % LEARN_INTERVAL == 0):
                mem_size = blue_fighter_models[0].get_memory_size()
                batch_indexes = random.sample(range(mem_size), BATCH_SIZE)
                # 采样
                # Blue
                blue_s_screen_batches, blue_s_info_batches, blue_mate_a_batches, blue_s__screen_batches, blue_s__info_batches = [], [], [], [], []
                blue_r_batches, blue_alive_batches, blue_done_batches, blue_self_a_batches = [], [], [], []
                for blue_agent in blue_fighter_models:
                    [s_screen_batch, s_info_batch, alive_batch, self_a_batch, r_batch,
                     s__screen_batch, s__info_batch, done_batch] = blue_agent.memory.sample_replay(batch_indexes)
                    self_a_batch = torch.unsqueeze(self_a_batch, 1)
                    blue_s_screen_batches.append(s_screen_batch)
                    blue_s_info_batches.append(s_info_batch)
                    blue_s__screen_batches.append(s__screen_batch)
                    blue_s__info_batches.append(s__info_batch)
                    blue_r_batches.append(r_batch)
                    blue_alive_batches.append(alive_batch)
                    blue_done_batches.append(done_batch)
                    blue_self_a_batches.append(self_a_batch)
                for i in range(len(blue_fighter_models)):
                    mate_agents = [j for j in range(len(blue_fighter_models)) if j != i]
                    for j in range(9):
                        if j == 0:
                            mate_a_batch = blue_self_a_batches[mate_agents[j]]
                        else:
                            mate_a_batch = torch.cat([mate_a_batch, blue_self_a_batches[mate_agents[j]]], 1)
                    blue_mate_a_batches.append(mate_a_batch)

                # Red
                red_s_screen_batches, red_s_info_batches, red_mate_a_batches, red_s__screen_batches, red_s__info_batches = [], [], [], [], []
                red_r_batches, red_alive_batches, red_done_batches, red_self_a_batches = [], [], [], []
                for red_agent in red_fighter_models:
                    [s_screen_batch, s_info_batch, alive_batch, self_a_batch, r_batch,
                     s__screen_batch, s__info_batch, done_batch] = red_agent.memory.sample_replay(batch_indexes)
                    self_a_batch = torch.unsqueeze(self_a_batch, 1)
                    red_s_screen_batches.append(s_screen_batch)
                    red_s_info_batches.append(s_info_batch)
                    red_s__screen_batches.append(s__screen_batch)
                    red_s__info_batches.append(s__info_batch)
                    red_r_batches.append(r_batch)
                    red_alive_batches.append(alive_batch)
                    red_done_batches.append(done_batch)
                    red_self_a_batches.append(self_a_batch)
                for i in range(len(red_fighter_models)):
                    mate_agents = [j for j in range(len(red_fighter_models)) if j != i]
                    for j in range(9):
                        if j == 0:
                            mate_a_batch = red_self_a_batches[mate_agents[j]]
                        else:
                            mate_a_batch = torch.cat([mate_a_batch, red_self_a_batches[mate_agents[j]]], 1)
                    red_mate_a_batches.append(mate_a_batch)

                # Blue
                blue_actor_loss = train_actor(blue_fighter_models, blue_attention, blue_policy_optimizer_fighters, batch_indexes,
                                              blue_s_screen_batches, blue_s_info_batches, blue_mate_a_batches, blue_s__screen_batches, blue_s__info_batches,
                                              blue_r_batches, blue_alive_batches, blue_done_batches, blue_self_a_batches, red_self_a_batches)
                blue_critic_loss = train_critic(blue_fighter_models, blue_attention, blue_critic_optimizer_fighters,
                                                batch_indexes, blue_s_screen_batches, blue_s_info_batches, blue_mate_a_batches,
                                                blue_s__screen_batches, blue_s__info_batches, blue_r_batches, blue_alive_batches,
                                                blue_done_batches, blue_self_a_batches, red_self_a_batches)

                # red
                for i in range(len(blue_fighter_models)):
                    if i == 0:
                        blue_a_batch = blue_self_a_batches[i]
                    else:
                        blue_a_batch = torch.cat([blue_a_batch, blue_self_a_batches[i]], 1)

                for y in range(red_fighter_num):
                    other_agents = [agent for i, agent in enumerate(red_fighter_models) if i != y]
                    red_fighter_models[y].learn(writer, batch_indexes, other_agents, blue_a_batch)

                # 训练过程保存
                writer.add_scalar(tag='blue_actor_loss', scalar_value=blue_actor_loss,
                                  global_step=learn_step_counter)
                writer.add_scalar(tag='blue_critic_loss', scalar_value=blue_critic_loss,
                                  global_step=learn_step_counter)
                learn_step_counter += 1

            # 当达到一个epoch最大步数，强制进入下一个epoch
            if step_cnt > MAX_STEP:
                writer.add_scalar(tag='blue_epoch_reward', scalar_value=blue_epoch_reward,
                                      global_step=x)
                writer.add_scalar(tag='red_epoch_reward', scalar_value=red_epoch_reward,
                                      global_step=x)
                writer.add_scalar(tag='win', scalar_value=win,
                                      global_step=x)
                writer.add_scalar(tag='blue_win_rate', scalar_value=blue_win_rate,
                                      global_step=x)
                writer.add_scalar(tag='red_win_rate', scalar_value=red_win_rate,
                                      global_step=x)
                # 保存模型
                if len(blue_epoch_rewards) > 10:
                    # Blue
                    epoch_rewards2 = np.sort(blue_epoch_rewards)
                    threshold = epoch_rewards2[-5]
                    if blue_epoch_reward > threshold:
                        # 保存最优模型
                        for i in range(len(blue_fighter_models)):
                            save_path = 'model/Battle/blue/%d' % i
                            blue_fighter_models[i].save_best_model(save_path, blue_epoch_reward)
                        torch.save(blue_attention.state_dict(), 'model/Battle/blue/Attention/attention_%s.pkl' % str(int(blue_epoch_reward)))
                        att_pkls = os.listdir('model/Battle/blue/Attention/')
                        if len(att_pkls) > 5:
                            rewards = []
                            for file in att_pkls:
                                reward = int(float(file[10:-4]))
                                rewards.append(reward)
                            index = np.argmin(rewards)
                            os.remove(os.path.join('model/Battle/blue/Attention/', att_pkls[index]))

                print("blue_epoch_reward: %.3f   red_epoch_reward: %.3f" % (blue_epoch_reward, red_epoch_reward))
                blue_epoch_rewards.append(blue_epoch_reward)
                red_epoch_rewards.append(red_epoch_reward)
                break

    writer.close()
