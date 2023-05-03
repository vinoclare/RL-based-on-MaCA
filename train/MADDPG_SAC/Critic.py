import torch
import torch.nn as nn


class NetFighterCritic(nn.Module):
    # 输入img(screen)、info信息、所有Agent的action
    # 输出价值q_value

    def __init__(self, agent_num):
        super(NetFighterCritic, self).__init__()
        # Encoder
        self.conv1 = nn.Sequential(  # batch * 2 * 100 * 100
            nn.Conv2d(
                in_channels=2,
                out_channels=6,
                kernel_size=5,
                stride=3,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(  # batch * 6 * 16 * 16
            nn.Conv2d(
                in_channels=6,
                out_channels=12,
                kernel_size=5,
                stride=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(  # batch * 12 * 6 * 6
            nn.Conv2d(
                in_channels=12,
                out_channels=32,
                kernel_size=3,
                stride=1,
            ),
            nn.ReLU(),
        )  # batch * 32 * 4 * 4
        self.img_layernorm = nn.LayerNorm(32 * 4 * 4)
        self.info_fc = nn.Sequential(  # batch * 3
            nn.Linear(3, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        self.action_fc = nn.Sequential(  # batch * (4 * agent_num)
            nn.Linear(4 * agent_num, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # Decoder
        self.feature_fc = nn.Sequential(  # 25 * 25 * 64 + 256 + 256
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.decision_fc = nn.Sequential(
            nn.Linear(128, 1),
        )

        # Encoder2
        self.conv1_ = nn.Sequential(  # batch * 2 * 100 * 100
            nn.Conv2d(
                in_channels=2,
                out_channels=6,
                kernel_size=5,
                stride=3,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2_ = nn.Sequential(  # batch * 6 * 16 * 16
            nn.Conv2d(
                in_channels=6,
                out_channels=12,
                kernel_size=5,
                stride=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv3_ = nn.Sequential(  # batch * 12 * 6 * 6
            nn.Conv2d(
                in_channels=12,
                out_channels=32,
                kernel_size=3,
                stride=1,
            ),
            nn.ReLU(),
        )  # batch * 32 * 4 * 4
        self.img_layernorm_ = nn.LayerNorm(32 * 4 * 4)
        self.info_fc_ = nn.Sequential(  # batch * 3
            nn.Linear(3, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        self.action_fc_ = nn.Sequential(  # batch * (4 * agent_num)
            nn.Linear(4 * agent_num, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # Decoder2
        self.feature_fc_ = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.decision_fc_ = nn.Sequential(
            nn.Linear(128, 1),
        )

    def encoding(self, img, info, act):
        # q1
        # encoder
        img_feature1 = self.conv1(img)
        img_feature2 = self.conv2(img_feature1)
        img_feature3 = self.conv3(img_feature2)
        img_feature4 = img_feature3.view(img_feature3.size(0), -1)
        img_feature5 = self.img_layernorm(img_feature4)
        info_feature = self.info_fc(info)
        action_feature = self.action_fc(act)
        e1 = torch.cat((img_feature5, info_feature.view(info_feature.size(0), -1),
                        action_feature.view(action_feature.size(0), -1)), dim=1)

        # q2
        # encoder
        img_feature1_ = self.conv1_(img)
        img_feature2_ = self.conv2_(img_feature1_)
        img_feature3_ = self.conv3_(img_feature2_)
        img_feature4_ = img_feature3_.view(img_feature3_.size(0), -1)
        img_feature5_ = self.img_layernorm_(img_feature4_)
        info_feature_ = self.info_fc_(info)
        action_feature_ = self.action_fc_(act)
        e2 = torch.cat((img_feature5_, info_feature_.view(info_feature_.size(0), -1),
                        action_feature_.view(action_feature_.size(0), -1)), dim=1)
        return e1, e2

    def decoding(self, attention1, attention2):
        # q1
        f1 = self.feature_fc(attention1)
        q1 = self.decision_fc(f1)

        # q2
        f2 = self.feature_fc_(attention2)
        q2 = self.decision_fc_(f2)

        return q1, q2
