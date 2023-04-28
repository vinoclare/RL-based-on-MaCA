import torch
import torch.nn as nn


class NetFighterQ(nn.Module):
    # 输入img(screen)、info信息、所有Agent的action
    # 输出价值q_value

    def __init__(self, agent_num):
        super(NetFighterQ, self).__init__()
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
            nn.ReLU(),
        )
        self.action_fc = nn.Sequential(  # batch * (4 * agent_num)
            nn.Linear(4 * agent_num, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.feature_fc = nn.Sequential(  # 25 * 25 * 64 + 256 + 256
            nn.Linear((25 * 25 * 8 + 64 + 128), 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        self.decision_fc = nn.Sequential(
            nn.Linear(256, 1),
        )

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
        q = self.decision_fc(feature)
        return q
