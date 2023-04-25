import torch
import torch.nn as nn


class NetFighterActor(nn.Module):
    # 决定航向与攻击类型的Actor网络
    # 输入img（screen）信息、info信息、agent存活信息
    # 输出航向（[0-359]离散值）、攻击类型(连续值)

    def __init__(self):
        super(NetFighterActor, self).__init__()
        self.conv = nn.Sequential(  # batch * 100 * 100 * 3
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
        self.info_fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.LayerNorm(64),
            nn.Tanh(),
        )
        self.feature_fc = nn.Sequential(
            nn.Linear((25 * 25 * 8 + 64), 256),
            nn.LayerNorm(256),
            nn.Tanh(),
        )
        self.decision_fc = nn.Sequential(
            nn.Linear(256, 4),
        )

    def forward(self, img, info):
        img_feature = self.conv(img)
        img_feature = img_feature.view(img_feature.size(0), -1)
        img_feature = self.img_layernorm(img_feature)
        info_feature = self.info_fc(info)
        combined = torch.cat((img_feature, info_feature.view(info_feature.size(0), -1)), dim=1)
        feature = self.feature_fc(combined)
        decision = self.decision_fc(feature)
        return decision
