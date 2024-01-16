#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 22:05:07 2021

@author: csluo
"""
import torch.nn as nn
import torch
from S_pool import Beran,Gine_G
import math
import torch.nn.functional as F
import numpy as np
from Beran import beran10

class MLP(nn.Module):
        def __init__(self, input_size, common_size):
            super(MLP, self).__init__()
            self.linear = nn.Sequential(
                nn.Linear(input_size, input_size//2),
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
                 mode='beran'
                 ):
        super(My_net, self).__init__()
        self.num_classes=num_classes
        self.divide=4
        self.mode=mode
        self.kernel_size=list(range(2,max_kernel_size))
        self._EPSILON=1e-7
        #learnable scalar
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
        # bias vectors for each dimension p, p=kernel_size
        real_disturb=[self.disturb2[i]*self.arange_list.cuda() for i in range(len(self.kernel_size))]
        
        # general poincare plot for each p, and is shifted by bias vectors 
        x1_dis=[(i(x).permute(0,2,1).view(-1,x.shape[-1]-self.kernel_size[j]+1, self.kernel_size[j],self.divide)+real_disturb[j]+self._EPSILON).permute(0,1,3,2) for j,i in enumerate(self.conv)]
        
        # unify vectors for each p 
        x1_norm=[i.norm(dim=-1,keepdim=True) for i in x1_dis]
        norm_x1=[i/j for i,j in zip(x1_dis, x1_norm)]
        
        # inner product each vector vs.  each vector
        x1_inner=[torch.einsum('ndsc,nlsc->nsdl', i,i).view(-1,i.shape[1],i.shape[1]) for i in norm_x1]#batch,divide,time,dimension
                
        # sobolev statistcs 
        if self.mode=='gine':
            all_G2=self.gine(x1_inner)
        if self.mode=='beran':
            #slow but adaptive for custom degree
# =============================================================================
#             G=[Beran(j, self.kernel_size[i], self.max_degree).h_z2()/j.shape[-1] for i,j in enumerate(x1_inner)] # /n
#             all_G2=torch.stack(G,dim=1)
# =============================================================================

            #more efficient only for max_degree=10；
            #and  (/(j.shape[-1]*(j.shape[-1]-1))) might be more suitable for processing different RRI lengths. 
            #The use of /j.shape[-1] is specifically to align with the definition of the Sobolev test statistics, 
            #aiming to normalize the variance of the coefficients of each spherical harmonics to 1 when the samples are
            #uniformly distributed on the sphere, thereby ensuring that the asymptotic null distribution conforms to
            #a chi-squared distribution.
            G2=[beran10(j,self.kernel_size[i]).sum(1)/j.shape[-1]  for i,j in enumerate(x1_inner) ] 
            all_G2=torch.stack(G2,dim=1)

        allG_Rho=all_G2.view(-1, self.fl)
        #batch norm
        allG_Rho=self.first_bn(allG_Rho)     
        out= self.MLP(allG_Rho)
        return F.log_softmax(out,dim=-1)

        
        

