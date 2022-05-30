# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')
def DAN(source, target, kernel_mul=2.0, kernel_num=3, fix_sigma=None):
    batch_size = int(source.size()[0])
    batch_size1 = int(target.size()[0])
    print("batchsize=",batch_size1)
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)

    loss1 = 0
    for s1 in range(batch_size):
        for s2 in range(s1+1, batch_size):
            t1, t2 = s1+batch_size, s2+batch_size
            loss1 += kernels[s1, s2] + kernels[t1, t2]
    loss1 = loss1 / float(batch_size * (batch_size - 1) / 2)

    loss2 = 0
    for s1 in range(batch_size):
        for s2 in range(batch_size):
            t1, t2 = s1+batch_size, s2+batch_size
            loss2 -= kernels[s1, t2] + kernels[s2, t1]
    loss2 = loss2 / float(batch_size * batch_size)
    return loss1 + loss2

import torch
import numpy as np

def guassian_kernel(source, target, kernel_mul=1.0, kernel_num=2, fix_sigma=None):

    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0) 
    
    total0 = total.unsqueeze(0).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2) 
    

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    

    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                  bandwidth_temp in bandwidth_list]

    return sum(kernel_val)
  
def mmd(source, target, kernel_mul=2, kernel_num=3, fix_sigma=None):
    batch_size1 = int(source.size()[0])
    batch_size2 = int(target.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, 	
                             	kernel_num=kernel_num, 	
                              fix_sigma=fix_sigma)

    M = np.zeros((batch_size1+batch_size2,batch_size1+batch_size2))
    for i in range(batch_size1+batch_size2):
        for j in range(batch_size1+batch_size2):
            if i<batch_size1 and j <batch_size1:
                M[i][j] = 1/(batch_size1**2)
            elif i>=batch_size1 and j>=batch_size1:
                M[i][j] = 1/(batch_size2**2)
            else:
                M[i][j] = -1/(batch_size1*batch_size2)
    M = torch.Tensor(M).to(DEVICE)
    loss = torch.trace(torch.mm(kernels,M))

    return loss
