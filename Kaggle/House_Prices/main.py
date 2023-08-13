import torch
import torch.nn as nn
import torch.functional as F
import pandas as pd
import numpy as np
import sys
from tqdm import tqdm
from torch.utils.data import random_split
from data_processing import *
import random
from network import *
import matplotlib.pyplot as plt
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
set_seed(20)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_ratio = 0.8
test_ratio = 0.2
train_num = num_train * train_ratio
test_num = num_train * test_ratio

traindata_all = torch.cat((train_features, train_labels), dim=1)
traindata, testdata = random_split(traindata_all, [train_ratio, test_ratio])  # subset object
train_indices = traindata.indices
train_features_subset = train_features[train_indices]
train_labels_subset = train_labels[train_indices]

test_indices = testdata.indices
test_features_subset = train_features[test_indices]
test_labels_subset = train_labels[test_indices]


model = MyNetwork(num_features).to(device)
num_epochs = 30
learning_rate = 0.003
batch_size = 64
weight_decay = 0.001
if __name__ == '__main__':
    target = 'train'
    if target == 'train':
        train_loss, test_loss = train(model, train_features_subset, train_labels_subset,
                                      test_features_subset, test_labels_subset, num_epochs, learning_rate,
                                      batch_size, train_num, test_num, weight_decay=weight_decay)

        print('train_loss', train_loss)
        print('test_loss', test_loss)
        plt.figure()
        plt.plot(train_loss, label='Train_loss')
        plt.plot(test_loss, label='Test_loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('N:{} L:{} B:{} W:{}'.format(num_epochs, learning_rate, batch_size, weight_decay))
        plt.show()
        torch.save(model, 'f_model.pth')

    else:
        model = torch.load('f_model.pth').to(device)
        y_pred = test(model, test_features) * M
        print("SalePrice:", y_pred)
        y_pred = y_pred.squeeze().tolist()
        data = pd.DataFrame({'Id': test_id, 'SalePrice': y_pred})
        data.to_csv('sample_submission.csv', index=False)