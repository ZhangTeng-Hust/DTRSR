from sklearn.tree import DecisionTreeRegressor
from TwoStageTrAdaBoostR2 import TwoStageTrAdaBoostR2
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import torch
import Result_evalute
import model_define

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def data_prepare(Task, name):
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

    return source_x, source_y, target_x, target_y


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


def F_Fc_JDASE(model, target_x, source_x):
    params = filter(lambda p: p.requires_grad, model.parameters())
    with torch.no_grad():
        source_list, prediction2 = model.forward(source_x)
        target_list, prediction5 = model.forward(target_x)
    s_x = source_list[0] 
    t_x = target_list[0]

    return s_x, t_x

if __name__ == '__main__':
    model_num = 1
    SeedResult = np.zeros((10, 4 * model_num))
    TaskALL = ['No.5-No.6','No.6-No.5','No.5-No.7','No.7-No.5','No.6-No.7','No.7-No.6']

    for Task in TaskALL:
        source_x, source_y, target_x, target_y = data_prepare(Task, Task)
        name = 'F_Fc_JDASE'
        target7 = model_define.TargetCNN().to(DEVICE)
        load_TargetCNN(target7, name, Task)
        s_x, t_x = F_Fc_JDASE(target7, target_x, source_x)   
        s_x = s_x.numpy()
        t_x = t_x.numpy()
        source_y = source_y.numpy()
        target_y = target_y.numpy()

        tsneoriginal = TSNE(n_components=2, init='pca', random_state=0)  
        original_source = tsneoriginal.fit_transform(s_x)
        s_x = original_source

        tsnetarget = TSNE(n_components=2, init='pca', random_state=0)   
        original_target = tsneoriginal.fit_transform(t_x)
        t_x = original_target

        for i in range(10):
            seedrecord = pd.read_csv('SeedIndex/' + Task + '_Seed.csv').values
            index1 = seedrecord[i, :] 
            index2 = np.delete(np.arange(168), index1)
            t_xseed = t_x[index1, :]
            t_yseed = target_y[index1]
            t_xtest = t_x[index2, :]
            t_ytest = target_y[index2]

            X = np.concatenate(( s_x, t_xseed))
            y = np.concatenate(( source_y, t_yseed))
            y = np.squeeze(y)

            sample_size = [len(s_x), len(t_xseed)]
            n_estimators = 100
            steps = 3
            fold = 2
            random_state = np.random.RandomState(10)
            regr_1 = TwoStageTrAdaBoostR2(DecisionTreeRegressor(max_depth=3, random_state=144), n_estimators=n_estimators,
                                      sample_size=sample_size, steps=steps, fold=fold, random_state=random_state)
            regr_1.fit(X, y)

            y_pred1 = regr_1.predict(t_xtest)
            print('Results of No', i, 'seed(', name, ')')
            result7 = Result_evalute.predict(t_ytest, y_pred1)
            SeedResult[i,:] = result7
        principle = pd.DataFrame(data=SeedResult)
        principle.to_csv('Revised_results/' + Task + 'TrAdaBoost.csv')


