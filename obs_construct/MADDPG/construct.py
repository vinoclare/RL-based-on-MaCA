import numpy as np
import copy

class ObsConstruct:
    def __init__(self, size_x, size_y, detector_num, fighter_num):
        self.battlefield_size_x = size_x
        self.battlefield_size_y = size_y
        self.detector_num = detector_num
        self.fighter_num = fighter_num
        self.img_obs_reduce_ratio = 10

    def obs_construct(self, obs_raw_dict):
        obs_dict = {}
        detector_obs_list = []
        fighter_obs_list = []
        detector_data_obs_list = obs_raw_dict['detector_obs_list']
        fighter_data_obs_list = obs_raw_dict['fighter_obs_list']
        joint_data_obs_dict = obs_raw_dict['joint_obs_dict']
        detector_img, fighter_img, joint_img = self.__get_img_obs(detector_data_obs_list, fighter_data_obs_list, joint_data_obs_dict)
        detector_data, fighter_data = self.__get_data_obs(detector_data_obs_list, fighter_data_obs_list, joint_data_obs_dict)
        alive_status = self.__get_alive_status(detector_data_obs_list, fighter_data_obs_list)

        # detector
        for x in range(self.detector_num):
            img_context = detector_img[x, :, :, :]
            img_context = np.concatenate((img_context, joint_img[0, :, :, :]), axis=2)
            data_context = detector_data[x, :]
            detector_obs_list.append({'info': copy.deepcopy(data_context), 'screen': copy.deepcopy(img_context),
                             'alive': alive_status[x][0]})
        # fighter
        for x in range(self.fighter_num):
            img_context = fighter_img[x, :, :, :]
            img_context = np.concatenate((img_context, joint_img[0, :, :, :]), axis=2)
            data_context = fighter_data[x, :]
            fighter_obs_list.append({'info': copy.deepcopy(data_context), 'screen': copy.deepcopy(img_context),
                             'alive': alive_status[x + self.detector_num][0]})
        obs_dict['detector'] = detector_obs_list
        obs_dict['fighter'] = fighter_obs_list
        return obs_dict

    def __get_alive_status(self,detector_data_obs_list,fighter_data_obs_list):
        # 获取fighter与detector的存活信息
        alive_status = np.full((self.detector_num+self.fighter_num,1),True)
        for x in range(self.detector_num):
            if not detector_data_obs_list[x]['alive']:
                alive_status[x][0] = False
        for x in range(self.fighter_num):
            if not fighter_data_obs_list[x]['alive']:
                alive_status[x+self.detector_num][0] = False
        return alive_status

    def __get_img_obs(self, detector_data_obs_list, fighter_data_obs_list, joint_data_obs_dict):
        # 视野范围
        img_obs_size_x = int(self.battlefield_size_y / self.img_obs_reduce_ratio)
        img_obs_size_y = int(self.battlefield_size_x / self.img_obs_reduce_ratio)
        # 初始化img（3 channels）
        detector_img = np.full((self.detector_num, img_obs_size_x, img_obs_size_y, 3), 0, dtype=np.int32)
        fighter_img = np.full((self.fighter_num, img_obs_size_x, img_obs_size_y, 3), 0, dtype=np.int32)
        # 联合img（2 channels）
        joint_img = np.full((1, img_obs_size_x, img_obs_size_y, 2), 0, dtype=np.int32)

        tmp_pos_obs = np.full((img_obs_size_x, img_obs_size_y), 0, dtype=np.int32)
        for x in range(self.detector_num):
            if not detector_data_obs_list[x]['alive']:
                continue
            self.__set_value_in_img(tmp_pos_obs, int(detector_data_obs_list[x]['pos_y'] / self.img_obs_reduce_ratio),
                                    int(detector_data_obs_list[x]['pos_x'] / self.img_obs_reduce_ratio), 1)
        for x in range(self.fighter_num):
            if not fighter_data_obs_list[x]['alive']:
                continue
            self.__set_value_in_img(tmp_pos_obs, int(fighter_data_obs_list[x]['pos_y'] / self.img_obs_reduce_ratio),
                                    int(fighter_data_obs_list[x]['pos_x'] / self.img_obs_reduce_ratio), 2)

        # detector img 填充
        for x in range(self.detector_num):
            # if not alive, skip
            if not detector_data_obs_list[x]['alive']:
                continue
            # channel 0 填充
            # 在雷达探测到的敌人位置填充敌人id
            for y in range(len(detector_data_obs_list[x]['r_visible_list'])):
                self.__set_value_in_img(detector_img[x, :, :, 0],
                                        int(detector_data_obs_list[x]['r_visible_list'][y][
                                                'pos_y'] / self.img_obs_reduce_ratio),
                                        int(detector_data_obs_list[x]['r_visible_list'][y][
                                                'pos_x'] / self.img_obs_reduce_ratio),
                                        detector_data_obs_list[x]['r_visible_list'][y]['id'])

            # channel 1 填充
            # 在雷达探测到的敌人位置填充敌人类型（detector: 1, fighter: 2）
            for y in range(len(detector_data_obs_list[x]['r_visible_list'])):
                self.__set_value_in_img(detector_img[x, :, :, 1],
                                        int(detector_data_obs_list[x]['r_visible_list'][y][
                                                'pos_y'] / self.img_obs_reduce_ratio),
                                        int(detector_data_obs_list[x]['r_visible_list'][y][
                                                'pos_x'] / self.img_obs_reduce_ratio),
                                        detector_data_obs_list[x]['r_visible_list'][y]['type'] + 1)

            # channel 2 填充
            # 在自己所在位置填充255，友方detector的位置填充1，友方fighter的位置填充2，阵亡单位不填充
            detector_img[x, :, :, 2] = copy.deepcopy(tmp_pos_obs)
            self.__set_value_in_img(detector_img[x, :, :, 2],
                                    int(detector_data_obs_list[x]['pos_y'] / self.img_obs_reduce_ratio),
                                    int(detector_data_obs_list[x]['pos_x'] / self.img_obs_reduce_ratio), 255)

        # Fighter img 填充
        for x in range(self.fighter_num):
            # if not alive, skip
            if not fighter_data_obs_list[x]['alive']:
                continue
            # channel 0 填充
            # 在雷达探测到的敌人位置填充敌人id
            for y in range(len(fighter_data_obs_list[x]['r_visible_list'])):
                self.__set_value_in_img(fighter_img[x, :, :, 0],
                                        int(fighter_data_obs_list[x]['r_visible_list'][y][
                                                'pos_y'] / self.img_obs_reduce_ratio),
                                        int(fighter_data_obs_list[x]['r_visible_list'][y][
                                                'pos_x'] / self.img_obs_reduce_ratio),
                                        fighter_data_obs_list[x]['r_visible_list'][y]['id'])

            # channel 1 填充
            # 在雷达探测到的敌人位置填充敌人类型（detector: 1, fighter: 2）
            for y in range(len(fighter_data_obs_list[x]['r_visible_list'])):
                self.__set_value_in_img(fighter_img[x, :, :, 1],
                                        int(fighter_data_obs_list[x]['r_visible_list'][y][
                                                'pos_y'] / self.img_obs_reduce_ratio),
                                        int(fighter_data_obs_list[x]['r_visible_list'][y][
                                                'pos_x'] / self.img_obs_reduce_ratio),
                                        fighter_data_obs_list[x]['r_visible_list'][y]['type'] + 1)

            # channel 2 填充
            # 在自己所在位置填充255，友方detector的位置填充1，友方fighter的位置填充2，阵亡单位不填充
            fighter_img[x, :, :, 2] = copy.deepcopy(tmp_pos_obs)
            self.__set_value_in_img(fighter_img[x, :, :, 2],
                                    int(fighter_data_obs_list[x]['pos_y'] / self.img_obs_reduce_ratio),
                                    int(fighter_data_obs_list[x]['pos_x'] / self.img_obs_reduce_ratio), 255)

        # 全局信息
        # Passive detection
        for x in range(len(joint_data_obs_dict['passive_detection_enemy_list'])):
            # channel 3 填充
            # 在敌人位置填充id
            self.__set_value_in_img(joint_img[0, :, :, 0], int(joint_data_obs_dict['passive_detection_enemy_list'][x]['pos_y'] / self.img_obs_reduce_ratio),
                                    int(joint_data_obs_dict['passive_detection_enemy_list'][x]['pos_x'] / self.img_obs_reduce_ratio),
                                    joint_data_obs_dict['passive_detection_enemy_list'][x]['id'])
            # channel 4 填充
            # 在敌人位置填充type（detector: 1, fighter: 2）
            self.__set_value_in_img(joint_img[0, :, :, 1], int(joint_data_obs_dict['passive_detection_enemy_list'][x]['pos_y'] / self.img_obs_reduce_ratio),
                                    int(joint_data_obs_dict['passive_detection_enemy_list'][x]['pos_x'] / self.img_obs_reduce_ratio),
                                    joint_data_obs_dict['passive_detection_enemy_list'][x]['type'] + 1)

        # detector
        for x in range(self.detector_num):
            for y in range(len(detector_data_obs_list[x]['r_visible_list'])):
                # 往channel 3中填充能够观测到的信息
                self.__set_value_in_img(joint_img[0, :, :, 0],
                                        int(detector_data_obs_list[x]['r_visible_list'][y]['pos_y'] / self.img_obs_reduce_ratio),
                                        int(detector_data_obs_list[x]['r_visible_list'][y]['pos_x'] / self.img_obs_reduce_ratio),
                                        detector_data_obs_list[x]['r_visible_list'][y]['id'])
                # 往channel 4中填充能够观测到的信息
                self.__set_value_in_img(joint_img[0, :, :, 1],
                                        int(detector_data_obs_list[x]['r_visible_list'][y]['pos_y'] / self.img_obs_reduce_ratio),
                                        int(detector_data_obs_list[x]['r_visible_list'][y]['pos_x'] / self.img_obs_reduce_ratio),
                                        detector_data_obs_list[x]['r_visible_list'][y]['type'] + 1)
        # fighter
        for x in range(self.fighter_num):
            for y in range(len(fighter_data_obs_list[x]['r_visible_list'])):
                #  往channel 3中填充能够观测到的信息
                self.__set_value_in_img(joint_img[0, :, :, 0],
                                        int(fighter_data_obs_list[x]['r_visible_list'][y][
                                                'pos_y'] / self.img_obs_reduce_ratio),
                                        int(fighter_data_obs_list[x]['r_visible_list'][y][
                                                'pos_x'] / self.img_obs_reduce_ratio),
                                        fighter_data_obs_list[x]['r_visible_list'][y]['id'])
                # 往channel 4中填充能够观测到的信息
                self.__set_value_in_img(joint_img[0, :, :, 1],
                                        int(fighter_data_obs_list[x]['r_visible_list'][y][
                                                'pos_y'] / self.img_obs_reduce_ratio),
                                        int(fighter_data_obs_list[x]['r_visible_list'][y][
                                                'pos_x'] / self.img_obs_reduce_ratio),
                                        fighter_data_obs_list[x]['r_visible_list'][y]['type'] + 1)
        return detector_img, fighter_img, joint_img

    def __set_value_in_img(self, img, pos_x, pos_y, value):
        """
        draw 3*3 rectangle in img
        :param img:
        :param pos_x:
        :param pos_y:
        :param value:
        :return:
        """
        img_obs_size_x = int(self.battlefield_size_y / self.img_obs_reduce_ratio)
        img_obs_size_y = int(self.battlefield_size_x / self.img_obs_reduce_ratio)
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

    def __get_data_obs(self, detector_data_obs_list, fighter_data_obs_list, joint_data_obs_dict):
        detector_data = np.full((self.detector_num, 1), -1, dtype=np.int32)
        fighter_data = np.full((self.fighter_num, 3), -1, dtype=np.int32)

        # Detector info
        for x in range(self.detector_num):
            if detector_data_obs_list[x]['alive']:
                detector_data[x,0] = detector_data_obs_list[x]['course']
        # Fighter info
        for x in range(self.fighter_num):
            if fighter_data_obs_list[x]['alive']:
                fighter_data[x, 0] = fighter_data_obs_list[x]['course']
                fighter_data[x, 1] = fighter_data_obs_list[x]['l_missile_left']
                fighter_data[x, 2] = fighter_data_obs_list[x]['s_missile_left']

        return detector_data, fighter_data
