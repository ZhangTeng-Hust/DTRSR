# -*- coding: utf-8 -*-
"""
Created on 14:11,2022/01/18
@author: ZhangTeng
"""
import numpy as np
import pandas as pd
import torch

import Result_evalute
import CDA  # conditional
import MDA  # marginal
import model_define
import Seed_Module  # Seed replacement

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def data_prepare(Task, name, i):
    c1 = np.load('data/c1/c1.npy')[2:316, 0:3, :].reshape(313, 3, 50000)
    c4 = np.load('data/c4/c4.npy')[2:316, 0:3, :].reshape(313, 3, 50000)
    c6 = np.load('data/c6/c6.npy')[2:316, 0:3, :].reshape(313, 3, 50000)
    Tool_1 = pd.read_csv("data/c1_wear.csv").values[2:316, 4].reshape(313, 1)
    Tool_4 = pd.read_csv("data/c4_wear.csv").values[2:316, 4].reshape(313, 1)
    Tool_6 = pd.read_csv("data/c6_wear.csv").values[2:316, 4].reshape(313, 1)

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
    seedrecord = pd.read_csv('SeedIndex/' + name + '_Seed.csv').values
    index1 = seedrecord[i, :] 
    index2 = np.delete(np.arange(313), index1)
    t_xseed = target_x[index1, :, :]
    t_yseed = target_y[index1, :]
    t_xtest = target_x[index2, :, :]
    t_ytest = target_y[index2, :]

    return source_x, source_y, t_xseed, t_yseed, t_xtest, t_ytest


def load_TargetCNN(model, tag, Task):
    pretrained_dict = torch.load("Pre-trained-model/" + Task + ".pth")

    new_dict = model.state_dict()
    pretrained_dict1 = {k: v for k, v in pretrained_dict.items() if k in new_dict}
    new_dict.update(pretrained_dict1)
    model.load_state_dict(new_dict)

    if tag == 'Finetune_FC' or tag == 'F_Fc_JDA' or tag == 'F_Fc_JDASE' or tag == 'F_Fc_MMD' or tag == 'F_Fc_CDA':
        namelist = ['predict.bias', 'predict.weight',
                    'fc4.bias', 'fc4.weight',
                    'fc3.bias', 'fc3.weight',
                    'fc2.bias', 'fc2.weight',
                    'fc1.bias', 'fc1.weight']
        for name, value in model.named_parameters():
            if name in namelist:
                value.requires_grad = True
            else:
                value.requires_grad = False
    else:
        namelist = []
        for name, value in model.named_parameters():
            if name not in namelist:
                value.requires_grad = True
            else:
                value.requires_grad = False
    print('Moldel loading finished')


def test_TargetCNN(model, t_xtest):
    with torch.no_grad():
        inter_x, ypre = model.forward(t_xtest)
    for i in range(len(inter_x)):
        inter_x[i] = inter_x[i].cpu().data.numpy()
    ypre = ypre.cpu().data.numpy()
    return inter_x, ypre


def test_TargetCNN_Seed(model, t_xtest):
    inter_x, ypre = model.forward_pre(t_xtest)
    for i in range(len(inter_x)):
        inter_x[i] = inter_x[i].cpu().data.numpy()
    ypre = ypre.cpu().data.numpy()
    return inter_x, ypre


def FinetuneFC(model, t_xseed, t_yseed, epoch, learning_rate, regularization):

    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=regularization)
    criterion = torch.nn.MSELoss()  
    
    for i in range(epoch):
        List_xtrain, prediction = model.forward(t_xseed)
        Loss = criterion(prediction, t_yseed)
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()

def F_Fc_MMD(name, model, t_xseed, t_yseed, t_xtest, epoch, learning_rate, regularization):
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=regularization)
    criterion = torch.nn.MSELoss()
    for i in range(epoch):
        train_list, prediction1 = model.forward(t_xseed)  
        test_list, prediction2 = model.forward(t_xtest)
        MMD = MDA.MMD_loss()
        s = test_list[0]
        t = train_list[0]  
        MMDloss = MMD.forward(s, t)
        Loss = 0.01*  criterion(prediction1, t_yseed) + 100* MMDloss
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()

def F_Fc_CDA(name, model, t_xseed, t_yseed, t_xtest, epoch, learning_rate, regularization):
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=regularization)
    criterion = torch.nn.MSELoss()
    for i in range(epoch):
        train_list, prediction1 = model.forward(t_xseed) 
        with torch.no_grad():
            test_list, prediction2 = model.forward(t_xtest)

        CDA_loss = CDA.forward(test_list, prediction2, train_list, t_yseed)
        Loss = 0.01 * criterion(prediction1, t_yseed) + 100 * CDA_loss
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()

def F_Fc_JDA(name, model, t_xseed, t_yseed, t_xtest, epoch, learning_rate, regularization):
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=regularization)
    criterion = torch.nn.MSELoss()
    for i in range(epoch):
        train_list, prediction1 = model.forward(t_xseed) 
        with torch.no_grad():
            test_list, prediction2 = model.forward(t_xtest)
        MMD = MDA.MMD_loss()
        s = test_list[0]
        t = train_list[0]  
        MMDloss = MMD.forward(s, t)
        CEOD_loss = CDA.forward(test_list, prediction2, train_list, t_yseed)
        Loss = 0.01 * criterion(prediction1, t_yseed) +  100* MMDloss + 100 * CEOD_loss
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()


def F_Fc_JDASE(model, t_xseed, t_yseed, t_xtest, source_x, source_y, epoch, learning_rate,
               regularization):
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=regularization)
    criterion = torch.nn.MSELoss()

    for i in range(epoch):
        t_seed_list, prediction1 = model.forward(t_xseed)  
        with torch.no_grad():
            t_test_list, prediction4 = model.forward(t_xtest) 

        t_seed = torch.cat((t_seed_list[0], t_yseed), 1) 
        with torch.no_grad():
            source_list, prediction2 = model.forward(source_x)
        s = torch.cat((source_list[0], source_y), 1)  

        new_middle_test, label_pred, Residual = Seed_Module.CLUSTER(s, t_seed)
       
        new_source_liketarget = torch.from_numpy(new_middle_test).to(DEVICE)  

        feartue = new_source_liketarget[:, 0:(np.shape(new_source_liketarget)[1] - 1)]
        label = new_source_liketarget[:, -1].reshape(np.shape(new_source_liketarget)[0], 1)

        new_source_list, prediction3 = model.forward_pre(feartue)
        CEOD_loss = CDA.forward(t_test_list, prediction4, t_seed_list, t_yseed)
        MMD = MDA.MMD_loss()
        MMDloss = MMD.forward(feartue, t_seed_list[0])
        Loss = criterion(prediction3, label) +  criterion(prediction1, t_yseed) + 100 * CEOD_loss + MMDloss
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()
        if i % 10 == 0:
            print(i, Loss.data)



if __name__ == '__main__':
    model_num = 7
    SeedResult = np.zeros((10, 4 * model_num))
    Task = 'C1-C4'
    for i in range(10):
        source_x, source_y, t_xseed, t_yseed, t_xtest, t_ytest = data_prepare(Task, Task, i)
        
        learning_rate = 2e-3
        regularization = 1e-4
        epoch = 100
         if 1:
             name = 'Pre_Direct'
             target1 = model_define.TargetCNN().to(DEVICE)
             load_TargetCNN(target1, name, Task)
        
             print('Results of No', i, 'seed(', name, '):')
             PD_Xtest, PD_ytest_pre = test_TargetCNN(target1, t_xtest) 
             result1 = Result_evalute.predict(t_ytest.cpu().data.numpy(), PD_ytest_pre)
             print('---------------------------')
        
         if 1:
             name = 'Retrain_All'
             target2 = model_define.TargetCNN().to(DEVICE)

             optimizer = torch.optim.Adam(target2.parameters(), lr=learning_rate, weight_decay=regularization)
             criterion = torch.nn.MSELoss()
             for epoch in range(num_epochs):
                 list_txseed, prediction = target2.forward(t_xseed)
                 Loss = criterion(prediction, t_yseed)
                 optimizer.zero_grad()
                 Loss.backward()
                 optimizer.step()
             RTA_Xtest, RTA_ytest_pre = test_TargetCNN(target2, t_xtest)
             print('Results of No', i, 'seed(', name, '):')
             result2 = Result_evalute.predict(t_ytest.cpu().data.numpy(), RTA_ytest_pre)  
             print('---------------------------')
             result = np.hstack((result1, result2))

         if 1:
             name = 'Finetune_Fc'
             target3 = model_define.TargetCNN().to(DEVICE)
             load_TargetCNN(target3, name, Task)

             FinetuneFC(target3, t_xseed, t_yseed, epoch, learning_rate, regularization)
             FFC_Xtest, FFC_ytest_pre = test_TargetCNN(target3, t_xtest)
             print('Results of No', i, 'seed(', name, '):')
             result3 = Result_evalute.predict(t_ytest.cpu().data.numpy(), FFC_ytest_pre)
             result = np.hstack((result, result3))
             print('---------------------------')

         if 1:
             name = 'F_Fc_MMD'
             target4 = model_define.TargetCNN().to(DEVICE)
             load_TargetCNN(target4, name, Task)

             F_Fc_MMD(name, target4, t_xseed, t_yseed, t_xtest, epoch, learning_rate, regularization)
             MMD_Xtest, MMD_ytest_pre = test_TargetCNN(target4, t_xtest) 
             print('Results of No', i, 'seed(', name, ')')
             result4 = Result_evalute.predict(t_ytest.cpu().data.numpy(), MMD_ytest_pre)
             result = np.hstack((result, result4))
             print('---------------------------')
        
         if 1:
             name = 'F_Fc_CDA'
             target5 = model_define.TargetCNN().to(DEVICE)
             load_TargetCNN(target5, name, Task)

             F_Fc_CDA(name, target5, t_xseed, t_yseed, t_xtest, epoch, learning_rate, regularization)
             CDA_Xtest, CDA_ytest_pre = test_TargetCNN(target5, t_xtest)  
             print('Results of No', i, 'seed(', name, ')')
             result5 = Result_evalute.predict(t_ytest.cpu().data.numpy(), CDA_ytest_pre)
             result = np.hstack((result, result5))
             print('---------------------------')
        
        if 1:
             name = 'F_Fc_JDA'
             target6 = model_define.TargetCNN().to(DEVICE)
             load_TargetCNN(target6, name, Task)

             F_Fc_JDA(name, target6, t_xseed, t_yseed, t_xtest, epoch, learning_rate, regularization)
             JDA_Xtest, JDA_ytest_pre = test_TargetCNN(target6, t_xtest)  
             print('Results of No', i, 'seed(', name, ')')
             result6 = Result_evalute.predict(t_ytest.cpu().data.numpy(), JDA_ytest_pre)
             result = np.hstack((result, result6))
             print('---------------------------')

        if 1:
            name = 'F_Fc_JDASE'
            target7 = model_define.TargetCNN().to(DEVICE)
            load_TargetCNN(target7, name, Task)

            F_Fc_JDASE(target7, t_xseed, t_yseed, t_xtest, source_x, source_y, epoch, learning_rate,
                       regularization)
            JDASE_Xtest, JDASE_ytest_pre = test_TargetCNN(target7, t_xtest) 
            print('Results of No', i, 'seed(', name, ')')
            result7 = Result_evalute.predict(t_ytest.cpu().data.numpy(), JDASE_ytest_pre)
            result = np.hstack((result, result7))
            print('---------------------------')

        SeedResult[i, :] = result7
        name = model_num * ['MAE', 'MAPE', 'RMSE', 'R2']
        principle = pd.DataFrame(columns=name, data=SeedResult)
        principle.to_csv('Result/' + Task + '_result.csv')
