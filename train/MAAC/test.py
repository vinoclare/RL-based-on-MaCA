from train.MAAC.Critic import NetFighterCritic as FC
import torch

img = torch.rand([10, 5, 100, 100])
info = torch.rand([10, 3])
action = torch.rand([10, 80])


def disable_gradients(module):
    for p in module.parameters():
        p.required_grad = False

model = FC(10)
# disable_gradients(model)
print('\n', model.parameters().__next__().required_grad)


a = torch.Tensor([2.])
b = torch.Tensor([3.])
a.requires_grad = True
b.requires_grad = True
list1 = []
list1.append(a)
list1.append(b)
list2 = [i * 2 for i in list1]
out1 = list2[0] * 2