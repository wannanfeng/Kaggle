import sys
import torch
import pandas as pd
import math

# README 有提示
torch.set_default_tensor_type(torch.FloatTensor)  # 将默认的张量数据类型设置为浮点型（FloatTensor）

train_data = pd.read_csv("train.csv")  # (1460, 81)
test_data = pd.read_csv("test.csv")  # (1459, 80)
test_id = test_data.iloc[:, 0].values

all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))  # (2919, 79)
# normalize
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index   # 选择非字符串的数据
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 每个数值特征的均值变为0，所以可以直接用0来替换缺失值
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 离散变量转为虚拟变量 给字符型变量用数值代替 类似one-hot
all_features = pd.get_dummies(all_features, dummy_na=True)  # (2919, 331)

num_features = all_features.shape[1]
num_train = train_data.shape[0]

train_features = torch.tensor(all_features[:num_train].values, dtype=torch.float)
test_features = torch.tensor(all_features[num_train:].values, dtype=torch.float)

train_labels = torch.tensor(train_data.iloc[:, -1].values, dtype=torch.float)  # 1460
min_label, _ = torch.min(train_labels, dim=0)
scientific_notation = int('{:e}'.format(min_label).split('e+')[-1])
M = math.pow(10, scientific_notation)
train_labels = train_labels / torch.tensor(M)
train_labels = train_labels.unsqueeze(1)
