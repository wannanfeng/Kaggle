import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from data_processing import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class MyNetwork(nn.Module):
    def __init__(self, feature):
        super(MyNetwork, self).__init__()
        self.Linear1 = nn.Linear(feature, 256)
        self.Linear2 = nn.Linear(256, 10)
        self.Linear3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = x.to(device)
        x = self.relu(self.Linear1(x))
        x = self.relu(self.Linear2(x))
        x = self.relu(self.Linear3(x))
        return x


class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()
        self.loss = nn.MSELoss()
    def forward(self, x, y):
        # y_pred = torch.max(x, torch.tensor(1.0))
        # loss = torch.sqrt(self.loss(y_pred.log(), y.log()))  # log_rmse meet error
        loss = torch.sqrt(self.loss(x, y))
        return loss

def train(model, trainfeature, trainlabel, testfeature, testlabel,
          num_epochs, learning_rate, batch_size, train_num, test_num, weight_decay=0.001):
    train_ls, test_ls = [], []
    rmse = Myloss()
    traindata = TensorDataset(trainfeature, trainlabel)
    testdata = TensorDataset(testfeature, testlabel)
    train_iter = DataLoader(traindata, batch_size, shuffle=True, drop_last=False)
    test_iter = DataLoader(testdata, batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # weight_decay = L2正则化的λ
    for epoch in tqdm(range(num_epochs)):
        model.train()
        trainLoss = 0
        testLoss = 0
        for i, (x, y) in enumerate(train_iter):
            x, y = x.to(device), y.to(device)
            o = model(x)
            l = rmse(o, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            trainLoss += l.item()
            if (i+1) % 2 == 0:
                print('Epoch:[{}/{}] Step:[{}/{}] Train_Loss:{}'.format(epoch+1, num_epochs, i+1, len(train_iter), l.item()))
        trainLoss = trainLoss / train_num
        train_ls.append(trainLoss)
        model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(test_iter):
                x, y = x.to(device), y.to(device)
                o = model(x)
                l = rmse(o, y)
                testLoss += l.item()
            testLoss = testLoss / test_num
            test_ls.append(testLoss)
    return train_ls, test_ls

def test(model, test_feature):
    y_pred = model(test_feature)
    return y_pred
