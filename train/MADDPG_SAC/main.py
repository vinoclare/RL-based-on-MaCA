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
import numpy as np
from agent.fix_rule_no_att.agent import Agent
from interface import Environment
import random
from train.MADDPG_SAC import MADDPG_SAC as MADDPG
from torch.utils.tensorboard import SummaryWriter
from common.Replay3 import Memory
from configuration.reward import GlobalVar as REWARD

MAP_PATH = os.path.join(root_path, 'maps/1000_1000_fighter10v10.map')

RENDER = True  # 是否渲染，渲染能加载出实时的训练画面，但是会降低训练速度
MAX_EPOCH = 150
BATCH_SIZE = 256
GAMMA = 0.99  # reward discount
TAU = 0.99
BETA = 0  # 边界惩罚discount
replace_target_iter = 50
MAX_STEP = 999  # 1个epoch内最大步数
LEARN_INTERVAL = 500  # 学习间隔
start_learn_epoch = 10  # 第10个epoch开始训练

# 网络学习率
actor_lr = 3e-4
critic_lr = 3e-4
q_lr = 3e-4

DETECTOR_NUM = 0
FIGHTER_NUM = 10
MAX_MEM_SIZE = 3e4  # 经验回放池最大容量
ATTACK_IND_NUM = (DETECTOR_NUM + FIGHTER_NUM) * 2 + 1  # 导弹攻击类型数量（不攻击+中程导弹攻击目标+远程导弹攻击目标）
RADAR_NUM = 10  # 雷达频点总数

# COURSE_NUM = 16
# ACTION_NUM = COURSE_NUM * ATTACK_IND_NUM

# 清除tensorboard文件
runs_path = os.path.join(root_path, 'runs/MADDPG_SAC')
if not os.path.exists(runs_path):
    os.makedirs(runs_path)
for file in os.listdir(runs_path):
    path = os.path.join(runs_path, file)
    os.remove(path)


def boundary_punish(all_pos, actions):
    # 对Agent试图越过边界的惩罚
    ps = []
    for i, (pos, act) in enumerate(zip(all_pos, actions)):
        ps.append(0)
        # 靠近上下边界的情况
        if pos[1] < 50 or pos[1] > 950:
            dist = pos[1] if pos[1] < 50 else 1000 - pos[1]  # agent距离边界的距离
            theta = math.pi / 180 * act[0]  # 弧度
            ps[i] -= np.abs(math.sin(theta)) * (50 - dist)
        # 靠近左右边界的情况
        if pos[0] < 50 or pos[0] > 950:
            dist = pos[0] if pos[0] < 50 else 1000 - pos[0]  # agent距离边界的距离
            theta = math.pi / 180 * act[0]  # 弧度
            ps[i] -= np.abs(math.cos(theta)) * (50 - dist)
    return ps


if __name__ == "__main__":
    # 红色方为fix rule no attack，蓝色方为MADDPG
    red_agent = Agent()

    # 双方的obs构建模块
    red_agent_obs_ind = red_agent.get_obs_ind()
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
    for y in range(blue_fighter_num):
        blue_fighter_model = MADDPG.RLFighter(name='blue_%d' % y, agent_num=(DETECTOR_NUM + FIGHTER_NUM) * 2,
                                              attack_num=ATTACK_IND_NUM, fighter_num=FIGHTER_NUM, radar_num=RADAR_NUM,
                                              max_memory_size=MAX_MEM_SIZE, replace_target_iter=replace_target_iter,
                                              actor_lr=actor_lr, critic_lr=critic_lr, q_lr=q_lr, reward_decay=GAMMA,
                                              tau=TAU, batch_size=BATCH_SIZE)
        blue_fighter_models.append(blue_fighter_model)
    red_agent.set_map_info(size_x, size_y, blue_detector_num, blue_fighter_num)

    writer = SummaryWriter('runs/MADDPG_SAC')
    red_action_replay = Memory(MAX_MEM_SIZE)

    for y in range(blue_fighter_num):
        if not os.path.exists('model/MADDPG_SAC/%d' % y):
            os.makedirs('model/MADDPG_SAC/%d' % y)

    # 训练循环
    for x in range(MAX_EPOCH):
        print("Epoch: %d" % (x + 1))
        step_cnt = 0
        env.reset()  # 重置环境
        blue_epoch_reward = 0  # 记录一个epoch内的蓝方平均reward
        while True:
            # 获取双方初始环境观测
            if step_cnt == 0:
                red_obs_list = []
                red_obs_dict, blue_obs_dict = env.get_obs()

            # 获取蓝色方行动
            blue_alive = []  # 蓝队全队存活信息
            blue_obs_list = []  # 蓝色方的全体环境观测信息
            # blue_poses = []  # 蓝队全体位置坐标
            blue_fighter_action = []  # 蓝色方所有agent的行动
            for y in range(blue_fighter_num):  # 可以不用for循环，直接矩阵计算
                true_action = np.array([0, 1, 0, 0], dtype=np.int32)
                tmp_img_obs = blue_obs_dict['fighter'][y]['screen']
                tmp_img_obs = tmp_img_obs.transpose(2, 0, 1)
                tmp_info_obs = blue_obs_dict['fighter'][y]['info']
                alive = 1 if blue_obs_dict['fighter'][y]['alive'] else 0
                blue_alive.append(alive)
                # blue_poses.append(blue_obs_dict['fighter'][y]['pos'])
                true_action = blue_fighter_models[y].choose_action(tmp_img_obs, tmp_info_obs)
                blue_obs_list.append({'screen': copy.deepcopy(tmp_img_obs), 'info': copy.deepcopy(tmp_info_obs)})
                blue_fighter_action.append(true_action)
            blue_fighter_action = np.array(blue_fighter_action)

            # step X 1
            red_detector_action, red_fighter_action = red_agent.get_action(red_obs_dict, step_cnt)
            env.step(red_detector_action, red_fighter_action, blue_detector_action, blue_fighter_action)
            step_cnt += 1
            blue_step_reward = 0

            # step X 9
            for i in range(9):
                # 获取红色方行动
                red_obs_dict, _ = env.get_obs()
                red_detector_action, red_fighter_action = red_agent.get_action(red_obs_dict, step_cnt)
                env.step(red_detector_action, red_fighter_action, blue_detector_action, blue_fighter_action)

                # 获取reward
                red_detector_reward, red_fighter_reward, red_game_reward, blue_detector_reward, \
                    blue_fighter_reward, blue_game_reward = env.get_reward()
                blue_step_reward += (blue_fighter_reward + blue_game_reward)

                # blue_boundary_punish = boundary_punish(blue_poses, blue_fighter_action)
                # blue_boundary_punish = [BETA * i for i in blue_boundary_punish]
                # blue_fighter_reward2 = blue_fighter_reward + blue_boundary_punish

                env.step(red_detector_action, red_fighter_action, blue_detector_action, blue_fighter_action)
            step_cnt += 9

            # 红色方fix_rule_no_attack的动作样式转换为与MADDPG一致
            red_fighter_action2 = []
            for action in red_fighter_action:
                tem_action = [action['course'], action['r_fre_point'], action['j_fre_point'], action['hit_target']]
                red_fighter_action2.append(tem_action)

            # 红色方agent位置以及存活数量
            red_poses = []
            red_alive = 0
            for i in range(FIGHTER_NUM):
                if red_obs_dict['fighter_obs_list'][i]['alive']:
                    red_poses.append(red_obs_dict['fighter_obs_list'][i]['pos_x'])
                    red_poses.append(red_obs_dict['fighter_obs_list'][i]['pos_y'])
                    red_alive += 1
                else:
                    red_poses.append(0)
                    red_poses.append(0)

            # 保存红色方经验
            red_action_replay.store_replay(red_fighter_action2)

            # 保存蓝色方replay
            _, blue_obs_dict = env.get_obs()
            for y in range(blue_fighter_num):
                tmp_img_obs = blue_obs_dict['fighter'][y]['screen']
                tmp_img_obs = tmp_img_obs.transpose(2, 0, 1)
                tmp_info_obs = blue_obs_dict['fighter'][y]['info']
                blue_obs_list_ = {'screen': copy.deepcopy(tmp_img_obs), 'info': copy.deepcopy(tmp_info_obs)}
                self_action = blue_fighter_action[y]
                done = 0
                if env.get_done() or step_cnt > MAX_STEP:
                    done = 1
                    if red_alive == 0:
                        blue_step_reward[y] += REWARD.reward_totally_win
                    elif red_alive < 3:
                        blue_step_reward[y] += REWARD.reward_win
                blue_fighter_models[y].store_replay(blue_obs_list[y], blue_alive[y], self_action,
                                                    blue_step_reward[y], blue_obs_list_, done)

            # 获取reward
            red_detector_reward, red_fighter_reward, red_game_reward, blue_detector_reward, \
                blue_fighter_reward, blue_game_reward = env.get_reward()
            blue_step_reward += (blue_fighter_reward + blue_game_reward)
            blue_epoch_reward += blue_step_reward.mean()
            print("epoch: %d  step: %d  avg_step_reward: %.3f" % (x + 1, step_cnt, blue_step_reward.mean() / 10))

            # 环境判定完成后（回合完毕），开始学习模型参数
            if env.get_done():
                # detector_model.learn()
                if x+1 > start_learn_epoch:
                    writer.add_scalar(tag='blue_avg_epoch_reward', scalar_value=blue_epoch_reward/step_cnt,
                                      global_step=x-10)
                print("avg_epoch_reward: %.3f" % (blue_epoch_reward/step_cnt))
                break
            # 未达到done但是达到了学习间隔时也学习模型参数
            # 100个step learn一次
            if x+1 > start_learn_epoch and step_cnt != 0 and (step_cnt % LEARN_INTERVAL == 0):
                # detector_model.learn()
                mem_size = blue_fighter_models[0].get_memory_size()
                batch_indexes = random.sample(range(mem_size), BATCH_SIZE)
                for y in range(blue_fighter_num):
                    other_agents = [agent for i, agent in enumerate(blue_fighter_models) if i != y]
                    blue_fighter_models[y].learn('model/MADDPG_SAC/%d' % y, writer, batch_indexes, other_agents,
                                                 red_action_replay)

            # 当达到一个epoch最大步数，强制进入下一个epoch
            if step_cnt > MAX_STEP:
                if x+1 > start_learn_epoch:
                    mem_size = blue_fighter_models[0].get_memory_size()
                    batch_indexes = random.sample(range(mem_size), BATCH_SIZE)
                    for y in range(blue_fighter_num):
                        other_agents = [agent for i, agent in enumerate(blue_fighter_models) if i != y]
                        blue_fighter_models[y].learn('model/MADDPG_SAC/%d' % y, writer, batch_indexes, other_agents,
                                                     red_action_replay)
                    writer.add_scalar(tag='blue_avg_epoch_reward', scalar_value=blue_epoch_reward/step_cnt,
                                      global_step=x-10)
                break

    writer.close()
