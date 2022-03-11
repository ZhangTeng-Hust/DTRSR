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

class SourceCNN(nn.Module):
    def __init__(self):
        super(SourceCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(3, 64, 5, stride=5),
            nn.MaxPool1d(2, stride=2, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 5, stride=5),
            nn.MaxPool1d(2, stride=2, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, 5, stride=5),
            nn.MaxPool1d(2, stride=2, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, stride=3),
            nn.MaxPool1d(2, stride=2, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.predict = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        source = self.fc1(x)
        x = self.relu(source)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        result = self.predict(x)

        return result


class TargetCNN(nn.Module):
    def __init__(self):
        super(TargetCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(3, 64, 5, stride=5),
            nn.MaxPool1d(2, stride=2, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 5, stride=5),
            nn.MaxPool1d(2, stride=2, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, 5, stride=5),
            nn.MaxPool1d(2, stride=2, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, stride=3),
            nn.MaxPool1d(2, stride=2, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.predict = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        inter_x1 = self.relu(x)
        x = self.fc2(inter_x1)
        inter_x2 = self.relu(x)
        x = self.fc3(inter_x2)
        inter_x3 = self.relu(x)
        x = self.fc4(inter_x3)
        inter_x4 = self.relu(x)
        result = self.predict(inter_x4)

        target_list = list([inter_x1, inter_x2, inter_x3, inter_x4])

        return target_list, result

    def forward_pre(self, x):
        x = self.fc2(x)
        inter_x2 = self.relu(x)
        x = self.fc3(inter_x2)
        inter_x3 = self.relu(x)
        x = self.fc4(inter_x3)
        inter_x4 = self.relu(x)
        result = self.predict(inter_x4)

        target_list = list([inter_x2, inter_x3, inter_x4])

        return target_list, result


if __name__ == '__main__':

    c1 = np.load('data/c1/c1.npy')[2:316, 0:3, :].reshape(313, 3, 50000)
    c4 = np.load('data/c4/c4.npy')[2:316, 0:3, :].reshape(313, 3, 50000)
    c6 = np.load('data/c6/c6.npy')[2:316, 0:3, :].reshape(313, 3, 50000)
    Tool_1 = pd.read_csv("data/c1_wear.csv").values[2:316, 4].reshape(313, 1)
    Tool_4 = pd.read_csv("data/c4_wear.csv").values[2:316, 4].reshape(313, 1)
    Tool_6 = pd.read_csv("data/c6_wear.csv").values[2:316, 4].reshape(313, 1)

    # TaskALL = ['C1-C4', 'C4-C1', 'C1-C6', 'C6-C1','C4-C6','C6-C4']
    TaskALL = ['C6-C1']
    for Task in TaskALL:
        if Task == 'C1-C4':
            source_x = torch.Tensor(c1).to(DEVICE)
            source_y = torch.Tensor(Tool_1).to(DEVICE)
            target_x = torch.Tensor(c4).to(DEVICE)
            target_y = torch.Tensor(Tool_4).to(DEVICE)
        elif Task == 'C1-C6':
            source_x = torch.Tensor(c1).to(DEVICE)
            source_y = torch.Tensor(Tool_1).to(DEVICE)
            target_x = torch.Tensor(c6).to(DEVICE)
            target_y = torch.Tensor(Tool_6).to(DEVICE)
        elif Task == 'C4-C6':
            source_x = torch.Tensor(c4).to(DEVICE)
            source_y = torch.Tensor(Tool_4).to(DEVICE)
            target_x = torch.Tensor(c6).to(DEVICE)
            target_y = torch.Tensor(Tool_6).to(DEVICE)
        elif Task == 'C4-C1':
            source_x = torch.Tensor(c4).to(DEVICE)
            source_y = torch.Tensor(Tool_4).to(DEVICE)
            target_x = torch.Tensor(c1).to(DEVICE)
            target_y = torch.Tensor(Tool_1).to(DEVICE)
        elif Task == 'C6-C1':
            source_x = torch.Tensor(c6).to(DEVICE)
            source_y = torch.Tensor(Tool_6).to(DEVICE)
            target_x = torch.Tensor(c1).to(DEVICE)
            target_y = torch.Tensor(Tool_1).to(DEVICE)
        elif Task == 'C6-C4':
            source_x = torch.Tensor(c6).to(DEVICE)
            source_y = torch.Tensor(Tool_6).to(DEVICE)
            target_x = torch.Tensor(c4).to(DEVICE)
            target_y = torch.Tensor(Tool_4).to(DEVICE)

        Source = SourceCNN().to(DEVICE)

        learning_rate = 0.02
        regularization = 1e-4
        num_epochs = 100
        Batch_size = 64

        optimizer = torch.optim.Adam(Source.parameters(), lr=learning_rate, weight_decay=regularization)

        criterion = torch.nn.MSELoss()
        torch_dataset = Data.TensorDataset(source_x, source_y)
        loader = Data.DataLoader(dataset=torch_dataset, batch_size=Batch_size, shuffle=True)

        loss_list = []
        accuracy_list = []
        for epoch in range(num_epochs):
            for step, (batch_x, batch_y) in enumerate(loader):
                prediction = Source.forward(batch_x)
                Loss = criterion(prediction, batch_y)
                loss_list.append(Loss.data/len(batch_x))   
                acc = acc_pre(batch_y.cpu().data.numpy(),prediction.cpu().data.numpy())
                accuracy_list.append(acc)
                optimizer.zero_grad()
                Loss.backward()
                optimizer.step()
            if epoch % 1 == 0:
                print(epoch, Loss.data)

        source_ypre = Source.forward(source_x)
        torch.cuda.empty_cache()
        target_ypre = Source.forward(target_x)
        print(Task)
        print('Results on source domain:')
        Result_evalute.predict(source_y.data.numpy(), source_ypre.data.numpy())
        print('Results on target domain:')
        Result_evalute.predict(target_y.data.numpy(), target_ypre.data.numpy())

        plt.scatter(source_y.data.numpy(),source_ypre.data.numpy(),c='r')
        plt.xlim([40, 150])
        plt.ylim([40, 150])
        plt.show()
        '''
        cycle = (np.arange(313)).reshape(313, 1)  
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
