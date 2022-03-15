# -*- coding: utf-8 -*-
"""
Created on 14:11,2021/09/13
@author: ZhangTeng
Build up the Pre-trained-model in different source data
"""
import numpy as np
import torch
import Result_evalute
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.utils.data as Data
from sklearn.metrics import r2_score
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def acc_pre(y_true,y_pred):
    R2 = r2_score(y_true,y_pred)
    return R2


class SourceCNN(nn.Module):
    def __init__(self):
        super(SourceCNN, self).__init__()
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
        inter_x = self.relu(x)

        x = self.fc2(inter_x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        result = self.predict(x)

        return result


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


if __name__ == '__main__':

    battery7 = pd.read_csv("data/7.csv", header=None).values.reshape(168, 1, 371)
    label7 = pd.read_csv("data/L7.csv", header=None).values
    battery6 = pd.read_csv("data/6.csv", header=None).values.reshape(168, 1, 371)
    label6 = pd.read_csv("data/L6.csv", header=None).values
    battery5 = pd.read_csv("data/5.csv", header=None).values.reshape(168, 1, 371)
    label5 = pd.read_csv("data/L5.csv", header=None).values

    # TaskALL = ['No.5-No.6', 'No.6-No.5', 'No.5-No.7', 'No.7-No.5','No.6-No.7','No.7-No.6']
    TaskALL = ['No.6-No.5']
    for Task in TaskALL:
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

        Source = SourceCNN().to(DEVICE)

        learning_rate = 0.005
        regularization = 1e-5
        num_epochs = 500

        optimizer = torch.optim.Adam(Source.parameters(), lr=learning_rate, weight_decay=regularization)

        criterion = torch.nn.MSELoss()


        loss_list = []
        accuracy_list = []
        for epoch in range(num_epochs):
            prediction = Source.forward(source_x)
            Loss = criterion(prediction, source_y)
            loss_list.append(Loss.data/len(source_x))   # 损失函数可视化
            acc = acc_pre(source_y.cpu().data.numpy(),prediction.cpu().data.numpy())
            accuracy_list.append(acc)
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()
            if epoch % 1 == 0:
                print(epoch, Loss.data)

        s_ypre = Source.forward(source_x)
        t_ypre = Source.forward(target_x)
        print(Task)
        print('Results on source domain:')
        Result_evalute.predict(source_y.data.numpy(), s_ypre.data.numpy())
        print('Results on target domain:')
        Result_evalute.predict(target_y.data.numpy(), t_ypre.data.numpy())

        plt.scatter(source_y.data.numpy(),s_ypre.data.numpy(),c='r')

        plt.show()
        '''
        cycle = (np.arange(313)).reshape(313, 1)  # 这里相当于定义了一个从1到313的横坐标
        figure = plt.figure()
        plt.title(Task, fontsize=15)
        plt.xlabel('Milling times', fontsize=13)
        plt.ylabel('Wear(μm)', fontsize=13)
        plt.plot(cycle, source_y.data.numpy(), label='source-True', c='r', lw=3)
        plt.plot(cycle, sypre.data.numpy(), label='source-prediction', c='C9', lw=1.8)
        plt.plot(cycle, target_y.data.numpy(), label='target-True', c='g', lw=3)
        plt.plot(cycle, typre.data.numpy(), label='target-prediction', lw=1.8, c='b')
        plt.legend(fontsize=12)
        plt.show()
        '''
        # plt.plot(loss_list,'b')
        # plt.show()
        # plt.plot(accuracy_list,'r')
        # plt.show()
        
        torch.save(Source.state_dict(), "Pre-trained-model/" + Task + "test.pth")
