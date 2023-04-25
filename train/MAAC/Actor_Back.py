import torch
import torch.nn as nn


class NetFighterActor(nn.Module):
    # 决定航向与攻击类型的Actor网络
    # 输入img（screen）信息、info信息、agent存活信息
    # 输出航向（[0-359]离散值）、攻击类型(连续值)

    def __init__(self, agent_num):
        super(NetFighterActor, self).__init__()

        self.img_nets = nn.ModuleList()
        self.info_nets = nn.ModuleList()
        self.decision_nets = nn.ModuleList()

        for i in range(agent_num):
            img_convs = nn.Sequential()
            info_fcs = nn.Sequential()
            decision_fcs = nn.Sequential()

            conv = nn.Sequential(  # batch * 100 * 100 * 3
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
            img_layernorm = nn.LayerNorm(25 * 25 * 8)
            info_fc = nn.Sequential(
                nn.Linear(3, 64),
                nn.LayerNorm(64),
                nn.Tanh(),
            )
            feature_fc = nn.Sequential(
                nn.Linear((25 * 25 * 8 + 64), 256),
                nn.LayerNorm(256),
                nn.Tanh(),
            )
            decision_fc = nn.Sequential(
                nn.Linear(256, 4),
            )

            img_convs.add_module('conv', conv)
            img_convs.add_module('img_layernorm', img_layernorm)
            info_fcs.add_module('info_fc', info_fc)
            decision_fcs.add_module('feature_fc', feature_fc)
            decision_fcs.add_module('decision_fc', decision_fc)

            self.img_nets.append(img_convs)
            self.info_nets.append(info_fcs)
            self.decision_nets.append(decision_fcs)

    def forward(self, imgs, infos):
        img_features = [img_net(img).view(imgs.shape[1], -1) for img, img_net in zip(imgs, self.img_nets)]
        info_features = [info_net(info) for info, info_net in zip(infos, self.info_nets)]
        combined_features = [torch.cat((img_fea, info_fea.view(info_fea.size(0), -1)), dim=1) for img_fea, info_fea in zip(img_features, info_features)]
        decisions = [dec_net(combine) for combine, dec_net in zip(combined_features, self.decision_nets)]

        return decisions
