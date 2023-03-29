import torch
import torch.nn as nn

class NetFighterActor(nn.Module):
    # 决定航向与攻击类型的Actor网络
    # 输入img（screen）信息、info信息、agent存活信息
    # 输出航向（[0-359]离散值）、攻击类型(连续值)

    def __init__(self):
        super(NetFighterActor, self).__init__()
        self.decision_fc = nn.Sequential(
            nn.Linear(5, 2),
        )

    def forward(self, feature):
        decision = self.decision_fc(feature)
        return decision


eval_net_actor, target_net_actor = NetFighterActor(), NetFighterActor()
# print(eval_net_actor.state_dict())

def copy_param(eval_net, target_net, tau):
    # 将eval网络中的参数复制到target网络中
    # tau: target网络参数保留率
    tem_dict = {}
    for param_tensor in eval_net.state_dict():
        tem_value = tau * target_net.state_dict()[param_tensor] + (1 - tau) * eval_net.state_dict()[param_tensor]
        tem_dict[param_tensor] = tem_value
    target_net.load_state_dict(tem_dict)

print()
for param_tensor in eval_net_actor.state_dict():
    print(eval_net_actor.state_dict()[param_tensor])
print()

for param_tensor in target_net_actor.state_dict():
    print(target_net_actor.state_dict()[param_tensor])
print()

copy_param(eval_net_actor, target_net_actor, 0)
for param_tensor in eval_net_actor.state_dict():
    print(eval_net_actor.state_dict()[param_tensor])
print()
for param_tensor in target_net_actor.state_dict():
    print(target_net_actor.state_dict()[param_tensor])

