import torch
import torch.nn as nn


class NetFighterActor(nn.Module):
    # 决定航向与攻击类型的Actor网络
    # 输入img（screen）信息、info信息、agent存活信息
    # 输出航向（[0-359]离散值）、攻击类型(连续值)

    def __init__(self):
        super(NetFighterActor, self).__init__()
        self.conv1 = nn.Sequential(  # batch * 2 * 100 * 100
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
        self.info_fc = nn.Sequential(
            nn.Linear(3, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        self.feature_fc = nn.Sequential(
            nn.Linear((25 * 25 * 6 + 64), 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )
        self.mean_head = nn.Sequential(
            nn.Linear(256, 4),
        )
        self.log_std_head = nn.Sequential(
            nn.Linear(256, 4),
        )

    def forward(self, img, info):
        img_feature1 = self.conv1(img)
        img_feature2 = self.conv2(img_feature1)
        img_feature3 = img_feature2.view(img_feature2.size(0), -1)
        img_feature4 = self.img_layernorm(img_feature3)
        info_feature = self.info_fc(info)
        combined = torch.cat((img_feature4, info_feature.view(info_feature.size(0), -1)), dim=1)
        feature = self.feature_fc(combined)
        means = self.mean_head(feature)
        log_stds = self.log_std_head(feature)
        log_stds = torch.clamp(log_stds, -20, 2)
        return means, log_stds
