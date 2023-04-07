import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class NetFighterCritic(nn.Module):
    # 输入img(screen)、info信息、所有Agent的action
    # 输出价值q_value

    def __init__(self, agent_num):
        super(NetFighterCritic, self).__init__()
        self.agent_num = agent_num

        self.info_encoders = nn.ModuleList()
        self.img_encoders = nn.ModuleList()
        self.info_encoders2 = nn.ModuleList()
        self.img_encoders2 = nn.ModuleList()
        self.critics = nn.ModuleList()

        # encoder部分
        for i in range(agent_num):
            img_encoder = nn.Sequential()
            img_conv = nn.Sequential(  # batch * 100 * 100 * 5
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
            img_flatten = nn.Flatten()
            img_layernorm = nn.LayerNorm(25 * 25 * 8)
            img_fc = nn.Sequential(
                nn.Linear(25 * 25 * 8, 256),
                nn.LayerNorm(256),
                nn.Tanh(),
            )

            img_encoder.add_module('img_conv', img_conv)
            img_encoder.add_module('img_flatten', img_flatten)
            img_encoder.add_module('img_layernorm', img_layernorm)
            img_encoder.add_module('img_fc', img_fc)
            self.img_encoders.append(img_encoder)

            img_encoder2 = nn.Sequential()
            img_conv2 = nn.Sequential(  # batch * 100 * 100 * 5
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
            img_flatten2 = nn.Flatten()
            img_layernorm2 = nn.LayerNorm(25 * 25 * 8)
            img_fc2 = nn.Sequential(
                nn.Linear(25 * 25 * 8, 256),
                nn.LayerNorm(256),
                nn.Tanh(),
            )

            img_encoder2.add_module('img_conv2', img_conv2)
            img_encoder2.add_module('img_flatten2', img_flatten2)
            img_encoder2.add_module('img_layernorm2', img_layernorm2)
            img_encoder2.add_module('img_fc2', img_fc2)
            self.img_encoders2.append(img_encoder2)

            info_encoder = nn.Sequential()
            fc1 = nn.Sequential(  # batch * (4 * agent_num)
                nn.Linear(4 * agent_num + 3, 128),
                nn.LayerNorm(128),
                nn.Tanh(),
            )
            encoder_fc = nn.Sequential(  # 25 * 25 * 64 + 256 + 256
                nn.Linear(128, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
            )
            info_encoder.add_module('fc1', fc1)
            info_encoder.add_module('encoder_fc', encoder_fc)
            self.info_encoders.append(info_encoder)

            encoder2 = nn.Sequential()
            fc2 = nn.Sequential(
                nn.Linear(3, 256),
                nn.LayerNorm(256),
                nn.Tanh(),
            )
            encoder2.add_module('fc2', fc2)
            self.info_encoders2.append(encoder2)

            # critic部分
            critic = nn.Sequential()
            critic.add_module('fc1', nn.Linear(1024 * 4 + 512, 256))
            critic.add_module('fc2', nn.LeakyReLU())
            critic.add_module('fc3', nn.Linear(256, 1))
            self.critics.append(critic)

        # Attention block
        # 4 heads
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        self.value_extractors = nn.ModuleList()
        for i in range(4):
            self.key_extractors.append(nn.Linear(512, 1024, bias=False))
            self.selector_extractors.append(nn.Linear(512, 1024, bias=False))
            self.value_extractors.append(nn.Sequential(nn.Linear(512, 1024),
                                                       nn.LeakyReLU()))

        # 共享参数的部分网络
        # self.shared_modules = [self.key_extractors, self.selector_extractors,
        #                        self.value_extractors, self.critic_encoders]

        self.decision_fc = nn.Sequential(
            nn.Linear(256, 1),
        )

    def shared_parameters(self):
        # 共享agent间部分网络的参数
        pass

    def forward(self, imgs, infos, acts, return_q=True, return_all_q=True, regularize=False, return_attend=False):
        combines = [torch.unsqueeze(torch.cat((info, act), dim=0), 0) for info, act in zip(infos, acts)]
        img_encodings = [img_encoder(torch.unsqueeze(img, 0)) for img, img_encoder in zip(imgs, self.img_encoders)]
        info_encodings = [info_encoder(combine) for combine, info_encoder in zip(combines, self.info_encoders)]
        all_encodings = [torch.cat((img_encoding, info_encoding), dim=1) for img_encoding, info_encoding in zip(img_encodings, info_encodings)]
        img_encodings2 = [img_encoder2(torch.unsqueeze(img, 0)) for img, img_encoder2 in zip(imgs, self.img_encoders2)]
        info_encodings2 = [info_encoder2(torch.unsqueeze(info, 0)) for info, info_encoder2 in zip(infos, self.info_encoders2)]
        s_encodings = [torch.cat((img_encoding2, info_encoding2), dim=1) for img_encoding2, info_encoding2 in zip(img_encodings2, info_encodings2)]

        all_head_keys = [[k_ext(enc) for enc in all_encodings] for k_ext in self.key_extractors]
        all_head_values = [[v_ext(enc) for enc in all_encodings] for v_ext in self.value_extractors]
        all_head_selectors = [[sel_ext(enc) for enc in s_encodings]
                              for sel_ext in self.selector_extractors]

        other_all_values = [[]for _ in range(self.agent_num)]
        all_attend_logits = [[] for _ in range(self.agent_num)]
        all_attend_probs = [[] for _ in range(self.agent_num)]

        # 为每个Agent计算Attention
        for head_keys, head_values, head_selectors in zip(
                all_head_keys, all_head_values, all_head_selectors):
            for i, selector in enumerate(head_selectors):
                keys = [k for j, k in enumerate(head_keys) if j != i]
                values = [v for j, v in enumerate(head_values) if j != i]

                # ???????????????????
                attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),
                                             torch.stack(keys).permute(1, 2, 0))
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
                attend_weights = F.softmax(scaled_attend_logits, dim=2)
                other_values = (torch.stack(values).permute(1, 2, 0) *
                                attend_weights).sum(dim=2)
                other_all_values[i].append(other_values)
                all_attend_logits[i].append(attend_logits)
                all_attend_probs[i].append(attend_weights)

        # 为每个Agent计算Q值
        all_rets = []
        for i in range(self.agent_num):
            critic_in = torch.cat((s_encodings[i], *other_all_values[i]), dim=1)
            q = self.critics[i](critic_in)

            all_rets.append(q)

        return all_rets
