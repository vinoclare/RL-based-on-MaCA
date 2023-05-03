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
            #nn.MaxPool2d(2),
        )  # batch * 32 * 4 * 4


    def forward(self, img):
        img_feature1 = self.conv1(img)
        img_feature2 = self.conv2(img_feature1)
        img_feature3 = self.conv3(img_feature2)

        return img_feature3


model = NetFighterActor()
input = torch.rand(1, 2, 100, 100)
out = model(input)
print(out.shape)