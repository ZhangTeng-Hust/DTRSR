"""
Created on 14:11,2021/12/10
@author: ZhangTeng
Conditional distribution adaptation
"""

import torch

def Gaussian_kernel(X, X2, gamma=0.4):

    X = torch.transpose(X, 1, 0)  
    X2 = torch.transpose(X2, 1, 0)
    n1, n2 = X.shape[1], X2.shape[1]
    n1sq = torch.sum(X ** 2, 0)
    n1sq = n1sq.float()
    n2sq = torch.sum(X2 ** 2, 0)
    n2sq = n2sq.float()
    D = torch.ones((n1, n2)).to('cuda') * n2sq + torch.transpose((torch.ones((n2, n1)).to('cuda') * n1sq), 1,
                                                                 0) + - 2 * torch.mm(
        torch.transpose(X, 1, 0), X2)
    K = torch.exp(-gamma * D)
    return K


def forward(X_p_list, Y_p, X_q_list, Y_q, lamda=1):
    sum = 0

    X_p = X_p_list
    X_q = X_q_list

    np = X_p.shape[0]
    nq = X_q.shape[0]
    I1 = torch.eye(np).to('cuda')
    I2 = torch.eye(nq).to('cuda')

    Kxpxp = Gaussian_kernel(X_p, X_p)
    Kxqxq = Gaussian_kernel(X_q, X_q)
    Kxqxp = Gaussian_kernel(X_q, X_p)
    Kypyq = Gaussian_kernel(Y_p, Y_q)
    Kyqyq = Gaussian_kernel(Y_q, Y_q)
    Kypyp = Gaussian_kernel(Y_p, Y_p)

    par1_a = torch.mm((torch.inverse(Kxpxp + np * lamda * I1)), Kypyp)
    par1_b = torch.mm(par1_a, (torch.inverse(Kxpxp + np * lamda * I1)))
    par1_c = torch.mm(par1_b, Kxpxp)
    sum1 = torch.trace(par1_c)

    par2_a = torch.mm((torch.inverse(Kxqxq + nq * lamda * I2)), Kyqyq)
    par2_b = torch.mm(par2_a, (torch.inverse(Kxqxq + nq * lamda * I2)))
    par2_c = torch.mm(par2_b, Kxqxq)
    sum2 = torch.trace(par2_c)

    par3_a = torch.mm((torch.inverse(Kxpxp + np * lamda * I1)), Kypyq)
    par3_b = torch.mm(par3_a, (torch.inverse(Kxqxq + nq * lamda * I2)))
    par3_c = torch.mm(par3_b, Kxqxp)
    sum3 = torch.trace(par3_c)
    sum += (sum1 + sum2 - 2 * sum3)
    return sum
