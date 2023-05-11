"""
一方为MADDPG,一方为fix-rule-no-attack的训练
"""

import os
import copy
import numpy as np
from agent.fix_rule_no_att.agent import Agent
from interface import Environment
import torch
from train.MADDPG import MADDPG_4 as MADDPG
from torch.utils.tensorboard import SummaryWriter

MAP_PATH = 'C:/MaCA/maps/1000_1000_fighter10v10.map'

RENDER = True  # 是否渲染，渲染能加载出实时的训练画面，但是会降低训练速度
MAX_EPOCH = 2000
BATCH_SIZE = 256
EPSILON = 0.9  # greedy policy
GAMMA = 0.99  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
DETECTOR_NUM = 0
FIGHTER_NUM = 10
ATTACK_IND_NUM = (DETECTOR_NUM + FIGHTER_NUM) * 2 + 1  # 导弹攻击类型数量（不攻击+中程导弹攻击目标+远程导弹攻击目标）
RADAR_NUM = 10  # 雷达频点总数

# COURSE_NUM = 16
# ACTION_NUM = COURSE_NUM * ATTACK_IND_NUM
LEARN_INTERVAL = 500  # 学习间隔（设置为1表示单步更新）
start_learn_threshold = 1000  # 当经验池积累5000条数据后才开始训练

# 清除tensorboard文件
for file in os.listdir('C:/MaCA/runs/single-4'):
    path = os.path.join('C:/MaCA/runs/single-4', file)
    os.remove(path)

if __name__ == "__main__":
    # 红色方为fix rule no attack，蓝色方为MADDPG
    red_agent = Agent()

    # 双方的obs构建模块
    red_agent_obs_ind = red_agent.get_obs_ind()
    blue_agent_obs_ind = 'MADDPG'

    # 创建环境
    os.chdir("C:/MaCA")
    env = Environment(MAP_PATH, red_agent_obs_ind, blue_agent_obs_ind, render=RENDER)
    # 获取环境信息
    size_x, size_y = env.get_map_size()
    red_detector_num, red_fighter_num, blue_detector_num, blue_fighter_num = env.get_unit_num()
    # 为双方设置环境信息
    blue_detector_action = []
    blue_fighter_model = MADDPG.RLFighter(name='blue', agent_num=(DETECTOR_NUM + FIGHTER_NUM) * 2, attack_num=ATTACK_IND_NUM,
                                          fighter_num=FIGHTER_NUM, radar_num=RADAR_NUM)
    red_agent.set_map_info(size_x, size_y, blue_detector_num, blue_fighter_num)
    # blue_fighter_model = MADDPG.RLFighter(ACTION_NUM, load_state=True, model_path='model/MADDPG/blue/model_000001000.pkl')  # 从已有参数中加载
    # red_fighter_model = MADDPG.RLFighter(ACTION_NUM, load_state=True, model_path='model/MADDPG/red/model_000001000.pkl')

    global_step_cnt = 0
    writer = SummaryWriter('runs/single-4')

    # 训练循环
    for x in range(MAX_EPOCH):
        print("Epoch: %d" % (x + 1))
        step_cnt = 0
        env.reset()  # 重置环境
        blue_epoch_reward = 0  # 记录一个epoch内的蓝方平均reward
        while True:
            red_obs_list = []
            # 获取双方初始环境观测
            if step_cnt == 0:
                red_obs_dict, blue_obs_dict = env.get_obs()
            # 获取红色方行动
            red_detector_action, red_fighter_action = red_agent.get_action(red_obs_dict, step_cnt)

            # 获取蓝色方行动
            blue_alive = []  # 蓝队全队存活信息
            blue_obs_list = []  # 蓝色方的全体环境观测信息
            blue_fighter_action = []  # 蓝色方所有agent的行动
            blue_obs_got_ind = [False] * blue_fighter_num  # 环境信息获取情况(True or False)
            for y in range(blue_fighter_num):  # 可以不用for循环，直接矩阵计算
                true_action = np.array([0, 1, 0, 0], dtype=np.int32)  # 第2、3维动作是预设定，未学习
                blue_obs_got_ind[y] = True
                tmp_img_obs = blue_obs_dict['fighter'][y]['screen']
                tmp_img_obs = tmp_img_obs.transpose(2, 0, 1)
                tmp_info_obs = blue_obs_dict['fighter'][y]['info']
                alive = 1 if blue_obs_dict['fighter'][y]['alive'] else 0
                blue_alive.append(alive)
                true_action = blue_fighter_model.choose_action(tmp_img_obs, tmp_info_obs, alive)
                blue_obs_list.append({'screen': copy.deepcopy(tmp_img_obs), 'info': copy.deepcopy(tmp_info_obs)})
                blue_fighter_action.append(true_action)
            blue_fighter_action = np.array(blue_fighter_action)

            # step
            # blue_fighter_action[0] = np.array([-89, 1, 0, 0])
            # print(blue_fighter_action[0])
            env.step(red_detector_action, red_fighter_action, blue_detector_action, blue_fighter_action)

            # 获取reward
            red_detector_reward, red_fighter_reward, red_game_reward, blue_detector_reward, \
                blue_fighter_reward, blue_game_reward = env.get_reward()
            # print('fighter_reward: %s  game_reward: %s' % (blue_fighter_reward, blue_game_reward))
            blue_step_reward = (blue_fighter_reward + blue_game_reward)

            step_cnt += 1

            for i in range(9):
                # 获取红色方行动
                env.step(red_detector_action, red_fighter_action, blue_detector_action, blue_fighter_action)

                # 获取reward
                red_detector_reward, red_fighter_reward, red_game_reward, blue_detector_reward, \
                    blue_fighter_reward, blue_game_reward = env.get_reward()
                blue_step_reward += (blue_fighter_reward + blue_game_reward)
                step_cnt += 1

            # 红色方fix_rule_no_attack的动作样式转换为与MADDPG一致
            red_fighter_action2 = []
            for action in red_fighter_action:
                tem_action = [action['course'], action['r_fre_point'], action['j_fre_point'], action['hit_target']]
                red_fighter_action2.append(tem_action)

            red_alive = 0
            for i in range(FIGHTER_NUM):
                if red_obs_dict['fighter_obs_list'][i]['alive']:
                    red_alive += 1

            # 保存replay
            red_obs_dict, blue_obs_dict = env.get_obs()
            for y in range(blue_fighter_num):
                tmp_img_obs = blue_obs_dict['fighter'][y]['screen']
                tmp_img_obs = tmp_img_obs.transpose(2, 0, 1)
                tmp_info_obs = blue_obs_dict['fighter'][y]['info']
                alive_ = int(blue_obs_dict['fighter'][y]['alive'])
                self_action = blue_fighter_action[y]
                mate_fighter_action = np.concatenate((blue_fighter_action[:y], blue_fighter_action[y + 1:]), 0)
                blue_obs_list_ = {'screen': copy.deepcopy(tmp_img_obs), 'info': copy.deepcopy(tmp_info_obs)}
                if env.get_done() or step_cnt > 2000:
                    if red_alive == 0:
                        blue_step_reward[y] += 100
                        print('epoch: %d  total win!' % x)
                    elif red_alive < 4:
                        blue_step_reward[y] += 50
                        print('epoch: %d  win!' % x)
                    blue_step_reward[y] += 3 * (10 - red_alive)

                blue_fighter_model.store_replay(blue_obs_list[y], blue_alive[y], alive_, self_action,
                                                mate_fighter_action, red_fighter_action2, blue_step_reward[y],
                                                blue_obs_list_)
            blue_epoch_reward += blue_step_reward.mean()
            # 未达到done但是达到了学习间隔时也学习模型参数
            # 100个step learn一次
            if (blue_fighter_model.get_memory_size() > start_learn_threshold) and (step_cnt % LEARN_INTERVAL == 0):
                # detector_model.learn()
                blue_fighter_model.learn('model/MADDPG/single-4', writer)

            # 环境判定完成后（回合完毕），开始学习模型参数
            if env.get_done() or step_cnt > 2000:
                writer.add_scalar(tag='blue_epoch_reward', scalar_value=blue_epoch_reward,
                                   global_step=x)
                break
            global_step_cnt += 1

    writer.close()