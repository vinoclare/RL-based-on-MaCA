import torch
import torch.nn as nn

class Attention(nn.Module):
    # Attention模块

    def __init__(self, agent_num):
        super(Attention, self).__init__()
        self.agent_num = agent_num
        self.key_extractor1 = nn.Linear(32 * 4 * 4 + 64 + 128, 256)
        self.key_extractor2 = nn.Linear(32 * 4 * 4 + 64 + 128, 256)

        self.value_extractor1 = nn.Linear(32 * 4 * 4 + 64 + 128, 256)
        self.value_extractor2 = nn.Linear(32 * 4 * 4 + 64 + 128, 256)

        self.query_extractor1 = nn.Linear(32 * 4 * 4 + 64 + 128, 256)
        self.query_extractor2 = nn.Linear(32 * 4 * 4 + 64 + 128, 256)

        self.softmax = nn.Softmax(dim=1)

    def scale_grads(self):
        # 梯度缩放
        for p in self.parameters():
            p.grad.data.mul_(1. / self.agent_num)

    def forward(self, encodings):
        k1, k2, v1, v2, q1, q2 = [], [], [], [], [], []
        for e in encodings:
            k1.append(self.key_extractor1(e))
            k2.append(self.key_extractor2(e))

            v1.append(self.value_extractor1(e))
            v2.append(self.value_extractor2(e))

            q1.append(self.query_extractor1(e))
            q2.append(self.query_extractor2(e))

        for i in range(len(encodings)):
            for j in range(len(encodings)):
                if j == 0:
                    tem_a1 = q1[i] * k1[j]
                    tem_a1 = torch.unsqueeze(tem_a1, 0)

                    tem_a2 = q2[i] * k2[j]
                    tem_a2 = torch.unsqueeze(tem_a2, 0)
                else:
                    tem_tem_a1 = q1[i] * k1[j]
                    tem_tem_a1 = torch.unsqueeze(tem_tem_a1, 0)
                    tem_a1 = torch.cat([tem_a1, tem_tem_a1], 0)

                    tem_tem_a2 = q2[i] * k2[j]
                    tem_tem_a2 = torch.unsqueeze(tem_tem_a2, 0)
                    tem_a2 = torch.cat([tem_a2, tem_tem_a2], 0)
            if i == 0:
                a1 = torch.unsqueeze(tem_a1, 0)
                a2 = torch.unsqueeze(tem_a2, 0)
            else:
                t_a1 = torch.unsqueeze(tem_a1, 0)
                a1 = torch.cat([a1, t_a1], 0)

                t_a2 = torch.unsqueeze(tem_a2, 0)
                a2 = torch.cat([a2, t_a2], 0)

        a1_ = self.softmax(a1)
        a2_ = self.softmax(a2)

        for i in range(len(encodings)):
            tem_a1 = a1_[i]
            tem_a2 = a2_[i]
            tem_b1 = 0
            tem_b2 = 0
            for j in range(len(encodings)):
                tem_b1 = tem_b1 + tem_a1[j] * v1[j]
                tem_b2 = tem_b2 + tem_a2[j] * v2[j]
            if i == 0:
                b1 = torch.unsqueeze(tem_b1, 0)
                b2 = torch.unsqueeze(tem_b2, 0)
            else:
                t_b1 = torch.unsqueeze(tem_b1, 0)
                t_b2 = torch.unsqueeze(tem_b2, 0)
                b1 = torch.cat([b1, t_b1], 0)
                b2 = torch.cat([b2, t_b2], 0)

        b = torch.cat([b1, b2])
        return b
