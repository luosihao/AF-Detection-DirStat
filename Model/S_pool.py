#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 15:44:33 2021

@author: csluo
"""

import torch

import torch.nn as nn
import math


class Beran(object):
    
    def __init__(self, x,p ,degree):
        self.x=x[:,torch.triu(torch.ones(x.shape[-1],x.shape[-1],dtype=torch.bool),diagonal=1).cuda() ]
# =============================================================================
#         self.x=x.flatten(1)
# =============================================================================

        self.alpha=p/2-1
        self.initial = [lambda x: torch.ones_like(x), lambda x: x*2*self.alpha]
        self.recurrence = lambda p1, p2, n,a, x: (torch.mul(x, p1)*2*(n+a-1)-(n+2*a-2)*p2)/n #p1 mean n-1
        self.degree=degree
        self.epsilon=1e-7
        if p!=2:
            self.coef=[1+i/self.alpha for i in range(degree+1)]
                                         
    def polynomial_generator(self):
        polys = [self.initial[0](self.x), self.initial[1](self.x)]
        if self.degree == 0:
            raise Exception('da mie')
        else:
            yield (self.coef[1] * polys[1]).sum(-1)
            for i in range(1, self.degree):
                new_poly = self.recurrence(polys[-1], polys[-2], i+1, self.alpha, self.x)
                polys.pop(0)
                polys.append(new_poly)
                yield (self.coef[i+1] * new_poly).sum(-1)
    
    def h_z2(self):
         batch=self.x.size(0)


         if self.alpha==0:
             temp=torch.acos(torch.clamp(self.x, -1 + self.epsilon, 1 - self.epsilon))
             sum_save=torch.zeros((batch),device=temp.device)
             for i in range(1, self.degree+1):
                  sum_save+=(2*torch.cos(i*temp)).sum(-1)
             # during inference, above can be subsitude into below without looping
# =============================================================================
#              temp=     2*torch.cos((self.degree+1)/2*temp)*torch.sin(self.degree/2*temp)/torch.sin(temp/2)
# =============================================================================
             return sum_save#torch.cat(h_z_sum,-1),h_z_all
         else :
             total = sum(self.polynomial_generator())

             return total

         
   


class Gine_G(nn.Module):
    def __init__(self,p ):
        super(Gine_G, self).__init__()
        self.p=p
        self.alpha=[i/2-1 for i in p]
        self.epsilon=1e-7    
        self.gamma_alpha=[math.pow(math.gamma(j+1/2)/math.gamma(j+1),2) for i,j in enumerate(self.alpha)]
        self.pi=math.pi
    def forward(self,x):
# =============================================================================
#         phi=torch.acos(torch.clamp(self.x, -1 + self.epsilon, 1 - self.epsilon))
# =============================================================================
        n=[i.shape[-1] for i in x]
        bool_matrixs=[torch.triu(torch.ones(i,i,dtype=torch.bool),diagonal=1).cuda() for i in n]
        up_triangle=[j[:,bool_matrixs[i]] for i,j in enumerate(x)]
# =============================================================================
#         sai=[acos(i) for i in up_triangle]
# =============================================================================
        sai=[torch.acos(torch.clamp(i, -1 + self.epsilon, 1 - self.epsilon)) for i in up_triangle]
# =============================================================================
#         Gn=[1/2*n[i]-1/n[i]*(torch.sin(j).sum(-1,keepdim=True))*self.gamma_alpha[i] for i,j in enumerate(sai)]
# =============================================================================
        Gn=[n[i]/2-(self.p[i]-1)/2/n[i]*self.gamma_alpha[i]*(torch.sin(j).sum(-1,keepdim=True)) for i,j in enumerate(sai)]
        Gn_tensor=torch.cat(Gn,dim=-1)
# =============================================================================
#         An=[n[i]/4-1/n[i]/self.pi*j.sum(-1,keepdim=True) for i,j in enumerate(sai)]
# =============================================================================
        An=[n[i]/4-1/n[i]/self.pi*j.sum(-1,keepdim=True) for i,j in enumerate(sai)]
        An_tensor=torch.cat(An,dim=-1)
     
# =============================================================================
#         return torch.cat([Gn_tensor, An_tensor],dim=-1)
# =============================================================================
        return Gn_tensor+An_tensor

        