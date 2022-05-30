import numpy as np
import torch
import Result_evalute
import torch.nn as nn
import pandas as pd

import TCA
import DAN

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')

class TargetCNN(nn.Module):
    def __init__(self):
        super(TargetCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, 5, stride=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, stride=3),
            nn.MaxPool1d(2, stride=2, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(768, 384)
        self.fc2 = nn.Linear(384, 64)
        self.fc3 = nn.Linear(64, 16)
        self.predict = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        inter_x1 = self.relu(x)
        x = self.fc2(inter_x1)
        inter_x2 = self.relu(x)
        x = self.fc3(inter_x2)
        inter_x3 = self.relu(x)
        result = self.predict(inter_x3)

        target_list = list([inter_x1, inter_x2, inter_x3])

        return target_list, result

    def forward_pre(self, x):
        x = self.fc2(x)
        inter_x2 = self.relu(x)
        x = self.fc3(inter_x2)
        inter_x3 = self.relu(x)
        result = self.predict(inter_x3)

        target_list = list([inter_x2, inter_x3])

        return target_list, result


Task = 'No.5-No.6'
battery7 = pd.read_csv("data/7.csv", header=None).values.reshape(168, 1, 371)
label7 = pd.read_csv("data/L7.csv", header=None).values
battery6 = pd.read_csv("data/6.csv", header=None).values.reshape(168, 1, 371)
label6 = pd.read_csv("data/L6.csv", header=None).values
battery5 = pd.read_csv("data/5.csv", header=None).values.reshape(168, 1, 371)
label5 = pd.read_csv("data/L5.csv", header=None).values

if Task == 'No.5-No.6':
    source_x = torch.Tensor(battery5).to(DEVICE)
    source_y = torch.Tensor(label5).to(DEVICE)
    target_x = torch.Tensor(battery6).to(DEVICE)
    target_y = torch.Tensor(label6).to(DEVICE)
elif Task == 'No.6-No.5':
    source_x = torch.Tensor(battery6).to(DEVICE)
    source_y = torch.Tensor(label6).to(DEVICE)
    target_x = torch.Tensor(battery5).to(DEVICE)
    target_y = torch.Tensor(label5).to(DEVICE)
elif Task == 'No.5-No.7':
    source_x = torch.Tensor(battery5).to(DEVICE)
    source_y = torch.Tensor(label5).to(DEVICE)
    target_x = torch.Tensor(battery7).to(DEVICE)
    target_y = torch.Tensor(label7).to(DEVICE)
elif Task == 'No.7-No.5':
    source_x = torch.Tensor(battery7).to(DEVICE)
    source_y = torch.Tensor(label7).to(DEVICE)
    target_x = torch.Tensor(battery5).to(DEVICE)
    target_y = torch.Tensor(label5).to(DEVICE)
elif Task == 'No.6-No.7':
    source_x = torch.Tensor(battery6).to(DEVICE)
    source_y = torch.Tensor(label6).to(DEVICE)
    target_x = torch.Tensor(battery7).to(DEVICE)
    target_y = torch.Tensor(label7).to(DEVICE)
elif Task == 'No.7-No.6':
    source_x = torch.Tensor(battery7).to(DEVICE)
    source_y = torch.Tensor(label7).to(DEVICE)
    target_x = torch.Tensor(battery6).to(DEVICE)
    target_y = torch.Tensor(label6).to(DEVICE)


seedrecord = pd.read_csv('SeedIndex/' + Task + '_Seed.csv').values

Source = TargetCNN()
pretrained_dict = torch.load("Pre-trained-model/"+Task+".pth")   # 载入模型
new_dict = Source.state_dict()
new_dict.update(pretrained_dict)
Source.load_state_dict(new_dict)


TCAResult = np.zeros((10,4))
DANResult = np.zeros((10,4))

# TCA方法
 for i in range(10):

     index1 = seedrecord[i, :]  # 选出第i行seedrecord中的数据
     index2 = np.delete(np.arange(167), index1)
     Xs = source_x
     Ys = source_y
     Xt = target_x[index2, :, :]
     Yt = target_y[index2, :]
     Xs_list, Ys_pre = Source.forward(Xs)
     Xt_list, Yt_pre = Source.forward(Xt)

     Xs = Xs_list[0].data.numpy()
     Xt = Xt_list[0].data.numpy()

     TCAmodel = TCA.TCA(kernel_type='rbf', dim=50, lamb=2, gamma=20)
     ypre = TCAmodel.fit_predict(Xs, Ys, Xt)
     result = Result_evalute.predict(Yt.data.numpy(), ypre)
     TCAResult[i, :] = result

     name = ['MAE', 'MAPE', 'RMSE', 'R2']
     principle = pd.DataFrame(columns=name, data=TCAResult)
     principle.to_csv('Result/' + Task + '-TCAresult.csv')

# DAN方法   无法自动循环，需要手动对研究的情况进行填写，写到对应的文件中。
learning_rate = 0.01
regularization = 1e-4
num_epochs = 100
Target = TargetCNN().to(DEVICE)
optimizer = torch.optim.Adam(Target.parameters(), lr=learning_rate, weight_decay=regularization)
criterion = torch.nn.MSELoss()

for q in range(10):

    index1 = seedrecord[q, :] 
    index2 = np.delete(np.arange(167), index1)
    Xs = source_x
    Ys = source_y
    Xt = target_x[index2, :, :]
    Yt = target_y[index2, :]
    Xt_s = target_x[index1, :, :]
    Yt_s = target_y[index1, :]

    for i in range(num_epochs):
        Xs_list, Ys_pre = Target.forward(Xs)
        Xt_s_list, Yt_s_pre = Target.forward(Xt_s)
        mmd_loss = DAN.mmd(Xs_list[0],Xt_s_list[0])

        Loss = criterion(Ys_pre, source_y) + mmd_loss
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()

        if (i % 10 == 0):
            print(i, Loss.data)
    Xt_list, Yt_pre = Target.forward(Xt)
    result2 = Result_evalute.predict(Yt.cpu().data.numpy(), Yt_pre.cpu().data.numpy())
    DANResult[q, :] = result2
    name = ['MAE', 'MAPE', 'RMSE', 'R2']
    principle2 = pd.DataFrame(columns=name, data=DANResult)
    principle2.to_csv('Result/' + Task + '-DANresult.csv')
