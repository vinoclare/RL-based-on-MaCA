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
from train.MADDPG_SAC import MADDPG_SAC as MADDPG
from torch.utils.tensorboard import SummaryWriter
from common.Replay3 import Memory
from configuration.reward import GlobalVar as REWARD

MAP_PATH = os.path.join(root_path, 'maps/1000_1000_fighter10v10.map')

RENDER = False  # 是否渲染，渲染能加载出实时的训练画面，但是会降低训练速度
PRERUN_EPOCH = 25

BATCH_SIZE = 256
GAMMA = 0.99  # reward discount
TAU = 0.99

replace_target_iter = 10  # target网络更新频率
MAX_STEP = 1999  # 1个epoch内最大步数
LEARN_INTERVAL = 500  # 学习间隔
start_learn_epoch = 5  # 第x个epoch开始训练
pass_step = 10  # 间隔x个step保存一次经验

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
                                              actor_lr=actor_lr, critic_lr=critic_lr, reward_decay=GAMMA,
                                              tau=TAU, batch_size=BATCH_SIZE)
        blue_fighter_models.append(blue_fighter_model)
    red_agent.set_map_info(size_x, size_y, blue_detector_num, blue_fighter_num)

    red_action_replay = Memory(MAX_MEM_SIZE)

    # 训练循环
    global_step_cnt = 0
    for x in range(PRERUN_EPOCH):
        print("Epoch: %d" % x)
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
            # 获取reward
            red_detector_reward, red_fighter_reward, red_game_reward, blue_detector_reward, \
                blue_fighter_reward, blue_game_reward = env.get_reward()
            blue_step_reward = (blue_fighter_reward + blue_game_reward)

            # step X pass_step-1
            for i in range(pass_step-1):
                # 获取红色方行动
                red_obs_dict, _ = env.get_obs()
                red_detector_action, red_fighter_action = red_agent.get_action(red_obs_dict, step_cnt)
                env.step(red_detector_action, red_fighter_action, blue_detector_action, blue_fighter_action)

                # 获取reward
                red_detector_reward, red_fighter_reward, red_game_reward, blue_detector_reward, \
                    blue_fighter_reward, blue_game_reward = env.get_reward()
                blue_step_reward += (blue_fighter_reward + blue_game_reward)

            step_cnt += pass_step - 1

            # 红色方fix_rule_no_attack的动作样式转换为与MADDPG一致
            red_fighter_action2 = []
            for action in red_fighter_action:
                tem_action = [action['course'], action['r_fre_point'], action['j_fre_point'], action['hit_target']]
                red_fighter_action2.append(tem_action)

            # 红色方agent位置以及存活数量
            red_poses = []
            red_alive = 0
            alive_id = []
            for i in range(FIGHTER_NUM):
                if red_obs_dict['fighter_obs_list'][i]['alive']:
                    red_poses.append(red_obs_dict['fighter_obs_list'][i]['pos_x'])
                    red_poses.append(red_obs_dict['fighter_obs_list'][i]['pos_y'])
                    red_alive += 1
                    alive_id.append(i+1)
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
                win = 0
                if env.get_done() or step_cnt > MAX_STEP:
                    done = 1
                    if red_alive == 0:
                        blue_step_reward[y] += REWARD.reward_totally_win
                        win = 2
                        print('epoch: %d  total win!' % x)
                    elif red_alive < 4:
                        blue_step_reward[y] += REWARD.reward_win
                        win = 1
                        print('epoch: %d  win!' % x)
                    blue_step_reward[y] += 30 * (10 - red_alive)
                blue_fighter_models[y].store_replay(blue_obs_list[y], blue_alive[y], self_action,
                                                    blue_step_reward[y]/pass_step, blue_obs_list_, done)
            global_step_cnt += 1
            blue_epoch_reward += blue_step_reward.mean()

            # 环境判定完成后（回合完毕），开始学习模型参数
            if env.get_done():
                break

            # 当达到一个epoch最大步数，强制进入下一个epoch
            if step_cnt > MAX_STEP:
                break

    if not os.path.exists('prerun_data'):
        os.makedirs('prerun_data')

    for i in range(len(blue_fighter_models)):
        path = os.path.join('prerun_data', '%d_data.npy' % i)
        blue_fighter_models[i].save_to_file(path)
    path = os.path.join('prerun_data', 'red_data.npy')
    red_action_replay.save_to_file(path)
