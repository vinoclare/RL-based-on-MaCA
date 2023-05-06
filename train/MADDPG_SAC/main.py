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
import torch
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
MAX_EPOCH = 2000
BATCH_SIZE = 256
GAMMA = 0.99  # reward discount
TAU = 0.99

replace_target_iter = 10  # target网络更新频率
MAX_STEP = 1999  # 1个epoch内最大步数
LEARN_INTERVAL = 500  # 学习间隔
pass_step = 10  # 间隔x个step保存一次经验
reward_scaling = 0.3  # reward缩放系数

# 网络学习率
actor_lr = 1e-4
critic_lr = 1e-4

DETECTOR_NUM = 0
FIGHTER_NUM = 10
MAX_MEM_SIZE = 1e4  # 经验回放池最大容量
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


def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
    seed_everything(1003)

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
                                              reward_scaling=reward_scaling, tau=TAU, batch_size=BATCH_SIZE)
        blue_fighter_models.append(blue_fighter_model)
    red_agent.set_map_info(size_x, size_y, blue_detector_num, blue_fighter_num)
    red_action_replay = Memory(MAX_MEM_SIZE)

    # 加载预存储的数据
    for i in range(len(blue_fighter_models)):
        path = os.path.join('prerun_data', '%d_data.npy' % i)
        blue_fighter_models[i].load_from_file(path)

    path = os.path.join('prerun_data', 'red_data.npy')
    red_action_replay.load_from_file(path)

    writer = SummaryWriter('runs/MADDPG_SAC')

    for y in range(blue_fighter_num):
        if not os.path.exists('model/MADDPG_SAC/%d/critic' % y):
            os.makedirs('model/MADDPG_SAC/%d/critic' % y)
        if not os.path.exists('model/MADDPG_SAC/%d/actor' % y):
            os.makedirs('model/MADDPG_SAC/%d/actor' % y)

    # 训练循环
    global_step_cnt = 0
    epoch_rewards = []
    for x in range(MAX_EPOCH):
        print("Epoch: %d" % x)
        step_cnt = 0
        env.reset()  # 重置环境
        blue_epoch_reward = 0  # 记录一个epoch内的蓝方平均reward
        while True:
            # 获取双方初始环境观测
            if step_cnt == 0:
                red_obs_list = []
                red_obs_dict, blue_obs_dict = env.get_obs()

            # 红色方agent位置
            red_poses = []
            for i in range(FIGHTER_NUM):
                if red_obs_dict['fighter_obs_list'][i]['alive']:
                    tem_dict = {'id': i + 1, 'pos_x': red_obs_dict['fighter_obs_list'][i]['pos_x'],
                                'pos_y': red_obs_dict['fighter_obs_list'][i]['pos_y']}
                    red_poses.append(tem_dict)

            # 获取蓝色方行动
            blue_alive = []  # 蓝队全队存活信息
            blue_obs_list = []  # 蓝色方的全体环境观测信息
            # blue_poses = []  # 蓝队全体位置坐标
            blue_fighter_action = []  # 蓝色方所有agent的行动
            actions = []
            for y in range(blue_fighter_num):
                true_action = np.array([0, 1, 0, 0], dtype=np.int32)
                tmp_img_obs = blue_obs_dict['fighter'][y]['screen']
                tmp_img_obs = tmp_img_obs.transpose(2, 0, 1)

                # 向图中添加全局信息
                for dic in red_poses:
                    set_value_in_img(tmp_img_obs[1], int(dic['pos_y']/10), int(dic['pos_x']/10), 90 + dic['id'] * 10)

                tmp_info_obs = blue_obs_dict['fighter'][y]['info']
                alive = 1 if blue_obs_dict['fighter'][y]['alive'] else 0
                blue_alive.append(alive)
                true_action, action = blue_fighter_models[y].choose_action(tmp_img_obs, tmp_info_obs)
                blue_obs_list.append({'screen': copy.deepcopy(tmp_img_obs), 'info': copy.deepcopy(tmp_info_obs)})
                blue_fighter_action.append(true_action)
                actions.append(action)

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
                step_cnt += 1

            # 红色方fix_rule_no_attack的动作样式转换为与MADDPG一致
            red_fighter_action2 = []
            for action in red_fighter_action:
                tem_action = [action['course'], action['r_fre_point'], action['j_fre_point'], action['hit_target']]
                reg_action = [tem_action[0]/180-1, tem_action[1]/RADAR_NUM*2-1, tem_action[2]/(RADAR_NUM-1)*2-1, tem_action[3]/10-1]
                red_fighter_action2.append(reg_action)

            # 红色方agent位置以及存活数量
            red_poses = []
            red_alive = 0
            for i in range(FIGHTER_NUM):
                if red_obs_dict['fighter_obs_list'][i]['alive']:
                    tem_dict = {'id': i + 1, 'pos_x': red_obs_dict['fighter_obs_list'][i]['pos_x'],
                                'pos_y': red_obs_dict['fighter_obs_list'][i]['pos_y']}
                    red_poses.append(tem_dict)
                    red_alive += 1

            # 保存红色方经验
            red_action_replay.store_replay(red_fighter_action2)

            # 保存蓝色方replay
            _, blue_obs_dict = env.get_obs()
            for y in range(blue_fighter_num):
                tmp_img_obs = blue_obs_dict['fighter'][y]['screen']
                tmp_img_obs = tmp_img_obs.transpose(2, 0, 1)

                for dic in red_poses:
                    set_value_in_img(tmp_img_obs[1], int(dic['pos_y']/10), int(dic['pos_x']/10), 90 + dic['id'] * 10)

                tmp_info_obs = blue_obs_dict['fighter'][y]['info']
                blue_obs_list_ = {'screen': copy.deepcopy(tmp_img_obs), 'info': copy.deepcopy(tmp_info_obs)}
                self_action = actions[y]
                done = 0
                win = 0
                if env.get_done() or step_cnt > MAX_STEP:
                    done = 1
                    if red_alive == 0:
                        blue_step_reward[y] += REWARD.reward_totally_win / 10
                        win = 2
                        print('epoch: %d  total win!' % x)
                    elif red_alive < 4:
                        blue_step_reward[y] += REWARD.reward_win / 10
                        win = 1
                        print('epoch: %d  win!' % x)
                    blue_step_reward[y] += 30 * (10 - red_alive) / 10
                blue_fighter_models[y].store_replay(blue_obs_list[y], blue_alive[y], self_action,
                                                    blue_step_reward[y]/pass_step, blue_obs_list_, done)
            global_step_cnt += 1
            # writer.add_scalar(tag='step_reward', scalar_value=blue_step_reward.mean(),
            #                   global_step=global_step_cnt)
            blue_epoch_reward += blue_step_reward.mean()

            # 环境判定完成后（回合完毕），开始学习模型参数
            if env.get_done():
                writer.add_scalar(tag='blue_epoch_reward', scalar_value=blue_epoch_reward,
                                  global_step=x)
                # 保存最优模型
                for i in range(len(blue_fighter_models)):
                    save_path = 'model/MADDPG_SAC/%d' % i
                    blue_fighter_models[i].save_best_model(save_path, epoch_rewards, blue_epoch_reward)
                epoch_rewards.append(blue_epoch_reward)
                print("avg_epoch_reward: %.3f" % (blue_epoch_reward/step_cnt))
                break
            # 未达到done但是达到了学习间隔时也学习模型参数
            if step_cnt != 0 and (step_cnt % LEARN_INTERVAL == 0):
                mem_size = blue_fighter_models[0].get_memory_size()
                batch_indexes = random.sample(range(mem_size), BATCH_SIZE)
                for y in range(blue_fighter_num):
                    other_agents = [agent for i, agent in enumerate(blue_fighter_models) if i != y]
                    blue_fighter_models[y].learn(writer, batch_indexes, other_agents,
                                                 red_action_replay)

            # 当达到一个epoch最大步数，强制进入下一个epoch
            if step_cnt > MAX_STEP:
                writer.add_scalar(tag='blue_epoch_reward', scalar_value=blue_epoch_reward,
                                      global_step=x)
                writer.add_scalar(tag='win', scalar_value=win,
                                      global_step=x)
                if len(epoch_rewards) > 10:
                    # 判断是否最优
                    epoch_rewards2 = np.sort(epoch_rewards)
                    threshold = epoch_rewards2[-5]
                    if blue_epoch_reward > threshold:
                        # 保存最优模型
                        for i in range(len(blue_fighter_models)):
                            save_path = 'model/MADDPG_SAC/%d' % i
                            blue_fighter_models[i].save_best_model(save_path, blue_epoch_reward)
                print("epoch_reward: %.3f" % blue_epoch_reward)
                epoch_rewards.append(blue_epoch_reward)
                break

    writer.close()
