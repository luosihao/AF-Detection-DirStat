import torch.nn as nn
import torch
from S_pool_fix import Beran
import torch.nn.functional as F

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
                
class map_radius(nn.Module):
    def __init__(self, conv_num,devide_inner):
        super(map_radius, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(conv_num,devide_inner))
        self.bias = nn.Parameter(torch.Tensor(conv_num,devide_inner))
        nn.init.kaiming_uniform_(self.weight)
        nn.init.kaiming_uniform_(self.bias)

    def forward(self, x):
        return  torch.sigmoid(x* self.weight+ self.bias)

class My_net(nn.Module):
    def __init__(self,
                 num_classes,
                 max_degree=10,
                 ):
        super(My_net, self).__init__()
        self.num_classes=num_classes
        # number of fix directions in each general poincare plot
        self.divide_inner=4
        # convolution num -> #of general poincare plot each of dimension p generate
        self.conv_num=2
        self.divide=self.conv_num* self.divide_inner
        # number of spherical convolution in each direction
        self.sphere_conv=2
        self.kernel_size=[2,3,9]
        self._EPSILON=1e-7
        feature_len=len(self.kernel_size)*self.divide**self.sphere_conv+ self.divide
        self.fl=feature_len
        self.MLP=MLP(feature_len,num_classes)   
        self.first_bn = nn.BatchNorm1d(feature_len,momentum=0.1)
        self.conv=nn.ModuleList([nn.Conv1d(in_channels=1,
                                           out_channels=i* self.conv_num,
                                           kernel_size=i+1,bias=False,dilation=1) for j,i in enumerate(self.kernel_size)]  )
        # radius mapping function
        self.simple_radius=map_radius( self.conv_num,self.divide_inner) 
        self.max_degree=max_degree
        # weight on each degree of gegenbauer coef, It corresponds to high-dimensional zonal spherical convolution. 
        self.weight_a=nn.Parameter(torch.ones(self.divide*len(self.kernel_size),max_degree,self.sphere_conv))
        #fix directions list
        self.bias_list=nn.ParameterList([nn.Parameter(torch.randn(i,self.divide))  for i in self.kernel_size])

    def forward(self, x):

        #unify fix direction 
        norm_direction=[i/i.norm(dim=0,keepdim=True) for i in self.bias_list]
        # general poincare
        x1_dis=[(i(x).permute(0,2,1).view(x.shape[0],-1, self.kernel_size[j],self.conv_num)+self._EPSILON).permute(0,1,3,2) for j,i in enumerate(self.conv)]
        #unify poincare to unit sphere 
        x1_norm=[i.norm(dim=-1,keepdim=True) for i in x1_dis]
        
        #The radius is mapped to a value between 0 and 1 through a straightforward linear transformation coupled with
        #a sigmoid activation, a technique possibly best suited for RRI signals. For other types of signals, 
        #this mapping process can be adjusted to utilize a multilayer perceptron (MLP). 
        #Only the dimension p=2 is required, as it represents the most informative radius.
        radius_mean =self.simple_radius(x1_norm[0]).mean(1).view(-1, self.divide)
            
        norm_x1=[(i/j).repeat(1,1,self.divide_inner,1) for i,j in zip(x1_dis, x1_norm)]
        x1_inner=[torch.einsum('ndsc,cs->nsd', i,j).reshape(-1,i.shape[1])
                  for i,j in zip(norm_x1,norm_direction)]
        
        
        G=[Beran(j,1, self.kernel_size[i], self.max_degree).h_z2()/  j.shape[-1]
           for i,j in enumerate(x1_inner)] # /n
            
        all_G2=torch.stack(G,dim=1)

        #self.weight_a can be regarded as the weighting for each degree of the Gegenbauer coef. 
        #It corresponds to high-dimensional zonal spherical convolution. 
        #For the three-dimensional case, you can refer to 
        #Figure 4 of  "Learning so (3) equivariant representations with spherical cnns" 2018: 52-68.
        #https://arxiv.org/pdf/1711.06721.pdf
        allG_Rho=all_G2.view(-1,self.divide*len(self.kernel_size),self.max_degree,1)*self.weight_a/self.weight_a.norm(dim=1,keepdim=True)
        allG_Rho=allG_Rho.sum(dim=-2).view(-1,self.fl-self.divide)
        
        # concatenation of the mean of the mapped radius and other Gegenbauer coef as features
        allG_Rho=torch.cat([radius_mean,allG_Rho],dim=1)
        
        allG_Rho=self.first_bn(allG_Rho)
        out= self.MLP(allG_Rho)
        return F.log_softmax(out,dim=-1)

        
