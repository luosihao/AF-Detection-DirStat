#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 22:05:07 2021

@author: csluo
"""
import torch.nn as nn
import torch
from S_pool import Beran,Gine_G
import torch.nn.parallel as parallel
import scipy.linalg as L
import math
import torch.nn.functional as F
import numpy as np

def comb(n,k):
    return math.factorial(n) // math.factorial(k) // math.factorial(n - k)
def dim_p_k(p,k):
    dpk=comb(p+k-3,p-2)+comb(p+k-2,p-2)
    return dpk
class MLP(nn.Module):
        def __init__(self, input_size, common_size):
            super(MLP, self).__init__()
            self.linear = nn.Sequential(
                nn.Linear(input_size, input_size//2),
# =============================================================================
#                 nn.Dropout(p=0.2),
# =============================================================================
                nn.ReLU(inplace=True),
                nn.Linear(input_size//2  , common_size)
            )
 
        def forward(self, x):
            out = self.linear(x)
            return out


class My_net(nn.Module):
    def __init__(self,
                 num_classes,
                 max_kernel_size=10,
                 max_degree=10,
                 mode='jupp'
                 ):
        super(My_net, self).__init__()
        self.num_classes=num_classes
        self.divide=4
        self.mode=mode
        self.kernel_size=list(range(2,max_kernel_size))
# =============================================================================
#         self.kernel_size=[max_kernel_size]
# =============================================================================

        self._EPSILON=1e-7

        self.disturb2=nn.Parameter(torch.zeros(len(self.kernel_size)) )
        if mode=='gine':
            self.gine=Gine_G(self.kernel_size)

            feature_len=self.divide*len(self.kernel_size)
        else:
            feature_len=self.divide*len(self.kernel_size)
        self.fl=feature_len
        self.MLP=MLP(feature_len,num_classes)   

        self.first_bn = nn.BatchNorm1d(feature_len,momentum=0.1)

        
        self.conv=nn.ModuleList([nn.Conv1d(in_channels=1,
                                           out_channels=i* self.divide,
                                           kernel_size=i,bias=False) for i in self.kernel_size]  )

        self.arange_list=torch.arange( self.divide)/self.divide 

        self.max_degree=max_degree


    def forward(self, x):

        real_disturb=[self.disturb2[i]*self.arange_list.cuda() for i in range(len(self.kernel_size))]
        
        
        x1_dis=[(i(x).permute(0,2,1).view(-1,x.shape[-1]-self.kernel_size[j]+1, self.kernel_size[j],self.divide)+real_disturb[j]+self._EPSILON).permute(0,1,3,2) for j,i in enumerate(self.conv)]
        x1_norm=[i.norm(dim=-1,keepdim=True) for i in x1_dis]
        norm_x1=[i/j for i,j in zip(x1_dis, x1_norm)]
        x1_inner=[torch.einsum('ndsc,nlsc->nsdl', i,i).view(-1,i.shape[1],i.shape[1]) for i in norm_x1]#batch,divide,time,dimension
                

        if self.mode=='gine':
            all_G2=self.gine(x1_inner)
            
     
        if self.mode=='jupp':

            G=[Beran(j, self.kernel_size[i], self.max_degree).h_z2()/j.shape[-1] for i,j in enumerate(x1_inner)] # /n

            all_G2=torch.stack(G,dim=1)




        allG_Rho=all_G2.view(-1, self.fl)
        allG_Rho=self.first_bn(allG_Rho)
        
        out= self.MLP(allG_Rho)


        return F.log_softmax(out,dim=-1)
# =============================================================================
#         return out
# =============================================================================
        
        
if __name__ == '__main__':
    import os
    from torchsummary import summary
    from thop import profile
    import util
    os.environ['CUDA_VISIBLE_DEVICES'] = "0" 

    model_ft=My_net(num_classes=2,max_kernel_size=10, max_degree=10)
# =============================================================================
#     flop,para=profile(model_ft.cuda(),torch.ones( 2,1, 100).cuda())
#     print("%.2fM"%(flop/1e6),"%.2fM"%(para/1e6))
#     print(util.get_parameter_number(model_ft))
# =============================================================================
    summary(model_ft.cuda(), ( 1, 100),device="cuda")
