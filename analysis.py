import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

data = pd.read_csv('att.csv')

x = data['x']
y = data['y']
# print(data.describe())
# plt.plot(x, y)
# plt.show()
y.hist()
plt.show()



# # Sigmoid函数
# x = np.linspace(-10, 10, 1000)
# # x = torch.tensor(x)
# # y = 1 / (1 + torch.exp(-x)) * 21
# y = [int(i) for i in x]
#
# plt.plot(x, y)
# plt.show()