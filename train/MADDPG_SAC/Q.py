import torch
import torch.nn as nn


class NetFighterQ(nn.Module):
    # 输入img(screen)、info信息、所有Agent的action
    # 输出价值q_value

    def __init__(self, agent_num):
        super(NetFighterQ, self).__init__()
        self.conv1 = nn.Sequential(  # batch *2 * 100 * 100
            nn.Conv2d(
                in_channels=2,
                out_channels=4,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(  # batch * 4 * 50 * 50
            nn.Conv2d(
                in_channels=4,
                out_channels=6,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )  # bacth * 6 * 25 * 25
        self.img_layernorm = nn.LayerNorm(25 * 25 * 6)
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
        self.feature_fc = nn.Sequential(  # 25 * 25 * 64 + 256 + 256
            nn.Linear((25 * 25 * 6 + 64 + 128), 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        self.decision_fc = nn.Sequential(
            nn.Linear(256, 1),
        )

    def forward(self, img, info, act):
        img_feature1 = self.conv1(img)
        img_feature2 = self.conv2(img_feature1)
        img_feature3 = img_feature2.view(img_feature2.size(0), -1)
        img_feature4 = self.img_layernorm(img_feature3)
        info_feature = self.info_fc(info)
        action_feature = self.action_fc(act)
        combined = torch.cat((img_feature4, info_feature.view(info_feature.size(0), -1),
                              action_feature.view(action_feature.size(0), -1)),
                             dim=1)
        feature = self.feature_fc(combined)
        q = self.decision_fc(feature)
        return q
