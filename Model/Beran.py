import torch
epsilon_1=1e-7
#using precalculated coef, r : weight for radius
def beran10(x,p,r=1):
    #we simplified the computation by only taking the values of the upper triangular matrix
    x=x[:,torch.triu(torch.ones(x.shape[1],x.shape[1],dtype=torch.bool),diagonal=1).cuda() ]
    if type(r)!=int:
        r=r[:,torch.triu(torch.ones(r.shape[1],r.shape[1],dtype=torch.bool),diagonal=1).cuda() ]
    if p==2:
        temp=torch.acos(torch.clamp(x, -1 + epsilon_1, 1 - epsilon_1))
        x=    2*torch.cos((10+1)/2*temp)*torch.sin(10/2*temp)/torch.sin(temp/2)
    if p==3:
        x=(969969*x**10)/256 + (230945*x**9)/128 - (2078505*x**8)/256 - (109395*x**7)/32 + (765765*x**6)/128 + (135135*x**5)/64 - (225225*x**4)/128 - (15015*x**3)/32 + (45045*x**2)/256 + (3465*x)/128 - 949/256
    if p==4:
        x=11264*x**10 + 5120*x**9 - 23040*x**8 - 9216*x**7 + 16128*x**6 + 5376*x**5 - 4480*x**4 - 1120*x**3 + 420*x**2 + 60*x - 7
    if p==5:
        x=(7436429*x**10)/256 + (1616615*x**9)/128 - (14549535*x**8)/256 - (692835*x**7)/32 + (4849845*x**6)/128 + (765765*x**5)/64 - (1276275*x**4)/128 - (75075*x**3)/32 + (225225*x**2)/256 + (15015*x)/128 - 3259/256
    if p==6:
        x=67584*x**10 + 28160*x**9 - 126720*x**8 - 46080*x**7 + 80640*x**6 + 24192*x**5 - 20160*x**4 - 4480*x**3 + 1680*x**2 + 210*x - 22
    if p==7:
        x=(37182145*x**10)/256 + (7436429*x**9)/128 - (66927861*x**8)/256 - (2909907*x**7)/32 + (20369349*x**6)/128 + (2909907*x**5)/64 - (4849845*x**4)/128 - (255255*x**3)/32 + (765765*x**2)/256 + (45045*x)/128 - 9265/256
    if p==8:
        x=292864*x**10 + 112640*x**9 - 506880*x**8 - 168960*x**7 + 295680*x**6 + 80640*x**5 - 67200*x**4 - 13440*x**3 + 5040*x**2 + 560*x - 57
    if p==9:
        x=(143416845*x**10)/256 + (26558675*x**9)/128 - (239028075*x**8)/256 - (9561123*x**7)/32 + (66927861*x**6)/128 + (8729721*x**5)/64 - (14549535*x**4)/128 - (692835*x**3)/32 + (2078505*x**2)/256 + (109395*x)/128 - 22135/256
    return x*r

#matlab code 
# =============================================================================
# syms x k
# Beran_dict=struct;
# p=3; %  S^p-1
# n=10;% max degree
# for p=3:10
#     for n=1:10
#         alpha=(p-2)/2;%Gegen
#         lambda=(p-3)/2;%Jacobi
#         d=p-1;
#         Beran=symsum((p-2+2*k)/(p-2)*gegenbauerC(k,alpha,x),k,1,n);
#         Beran_dict.(['dim' num2str(p)]).(['degree' num2str(n)])=Beran;
#     end
# end
# =============================================================================
