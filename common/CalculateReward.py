import pandas as pd

data = pd.read_csv('C:/Users/32137/Downloads/csv.csv')
reward = data['Value']
print(reward.mean())