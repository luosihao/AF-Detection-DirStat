import torch
class Beran(object):
    def __init__(self, x,y,p ,degree):

        self.weight=y
        self.x=x
        self.alpha=p/2-1
        self.initial = [lambda x: torch.ones_like(x), lambda x: x*2*self.alpha]
        self.recurrence = lambda p1, p2, n,a, x: (torch.mul(x, p1)*2*(n+a-1)-(n+2*a-2)*p2)/n #p1 mean n-1
        self.degree=degree
        self.epsilon=1e-7
        if p!=2:
            self.coef=[1+i/self.alpha for i in range(degree+1)]
            
    def h_z2(self):
         if self.alpha==0:
             temp=torch.acos(torch.clamp(self.x, -1 + self.epsilon, 1 - self.epsilon))
             sum_save=[]
             for i in range(1, self.degree+1):
                  sum_save.append((2*torch.cos(i*temp)*self.weight).sum(-1))
             return torch.stack(sum_save,dim=1)#torch.cat(h_z_sum,-1),h_z_all
         else :
             polys = [self.initial[0](self.x), self.initial[1](self.x)]
             sum_save=[ (self.coef[1] * polys[1]*self.weight).sum(-1)]
             for i in range(1, self.degree):
                 new_poly = self.recurrence(polys[-1], polys[-2], i+1, self.alpha, self.x)
                 polys.pop(0)
                 polys.append(new_poly)
                 sum_save.append( (self.coef[i+1] * new_poly*self.weight).sum(-1))
             return torch.stack(sum_save,dim=1)
