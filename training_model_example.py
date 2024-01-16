import os
from scipy import io
import csv
import numpy as np
import torch.utils.data as DATA
from  My_e2e import My_net
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score,confusion_matrix
import pandas as pd
import util
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter   
writer = SummaryWriter('/home/csluo/UVAF/log_weight_new/')

os.environ['CUDA_VISIBLE_DEVICES'] = "0" 
mode='jupp'
root_dir = "/data/csluo/alldatabase/data/"#
fs=1000
for mode in ['beran']:
    for d_name in ['UVAF']:
       
        save_root_path='/home/csluo/UVAF/new_model_weight_'+mode+'/'+d_name
        util.Mkdir(save_root_path)
        batch_size=10240
        num_epochs=100
        Data=[]
        Label=[]
        #load data
        d = torch.load(os.path.join(root_dir,d_name+'.pt'))
        for i in range(len(d)):
            data_inner=d[i]
            id_data=data_inner['id']
            label=data_inner['label']
            datainner=data_inner['rr']
            Data.append(datainner/fs)
            label[label!=1]=0
            Label.append(label)        
        
        # =============================================================================
        # classes = ['A', 'N', 'O', '~']
        # =============================================================================
        classes = ['N', 'A']
        classes_ind=range(len(classes)-1)
        classes_dict=dict(zip(classes,classes_ind))
        Ntrain = len(Data) # number of recordings on training set
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        criterion = nn.NLLLoss()
        np.random.seed(1)
        excel_epoch = pd.ExcelWriter(save_root_path+"/epoch_loss.xlsx", mode='w', engine='openpyxl')
        excel_best=pd.ExcelWriter(save_root_path+"/best_confusion.xlsx", mode='w', engine='openpyxl')
        excel_last=pd.ExcelWriter(save_root_path+"/last_confusion.xlsx", mode='w', engine='openpyxl')  
           
        kf = KFold( n_splits=5, shuffle=True, random_state=1)
        indexs=list(range(Ntrain))
        for k,(idxtrain,idxval) in enumerate(kf.split(indexs)):
             model =My_net(2,  max_kernel_size=10, max_degree=10,mode=mode) 
             if k==0:
                 print(util.get_parameter_number(model))
             model_inner=model.to(device)  
             optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
             print("Cross-validation run %d"%(k+1))   
            
             # since it is inter-patient ,so, you have to load separate records in each fold
             ytrain=[Label[i] for i in idxtrain]
             ytrain=np.concatenate(ytrain)
             xtrain=[Data[i] for i in idxtrain]
             xtrain=np.concatenate(xtrain)/1.0
             
             yval=[Label[i] for i in idxval]
             yval=np.concatenate(yval)
             xval=[Data[i] for i in idxval]
             xval=np.concatenate(xval)/1.0
             
             xtrain=torch.tensor(xtrain,dtype=torch.float32).unsqueeze(1)
             ytrain=torch.tensor(ytrain,dtype=torch.long)
             xval=torch.tensor(xval,dtype=torch.float32).unsqueeze(1)
             yval=torch.tensor(yval,dtype=torch.long)

             torch_dataset = DATA.TensorDataset(xtrain, ytrain)
             loader = DATA.DataLoader(
                 dataset=torch_dataset,      
                 batch_size=batch_size,      
                 shuffle=True,               
                 num_workers=4,             
                 persistent_workers=True,
                                          pin_memory=True,
                                          prefetch_factor=128,
                                          )
                 
             torch_dataset_val = DATA.TensorDataset(xval, yval)
             loader_val = DATA.DataLoader(
                 dataset=torch_dataset_val,     
                 batch_size=batch_size,      
                 shuffle=False,              
                 num_workers=4,             
                 )
             loader_dict={'train':loader,'val':loader_val}
             val_acc_history = []
             best_loss =0
             for epoch in range(num_epochs):
                 temp=[]
                 for phase in ['train', 'val']:
                    if phase == 'train':
                        model_inner.train()  # Set model to training mode 
                    else:
                        model_inner.eval()   # Set model to evaluate mode
                    with tqdm(loader_dict[phase], unit="batch", ncols=50, disable=True) as tepoch:
                         
                         pred_save=np.empty(shape=[0])
                         label_save=np.empty(shape=[0])
                         running_loss = 0.0   
                         running_corrects = 0
                         count=0
                         
                         for inputs, labels in tepoch:
                             inputs = inputs.to(device)
                             labels = labels.to(device)                              
                             if phase == 'train':
                                 optimizer.zero_grad()
                                 outputs= model_inner(inputs)
                                 loss = criterion(outputs, labels)
                                 loss_real=loss
                                 loss_real.backward()
                                 if False:
                                       for name, parms in model_inner.named_parameters(): 
                                            print('-->name:', name, '-->grad_requirs:',parms.requires_grad, \
                                             ' -->grad_value:',parms.grad)
                                 optimizer.step()
                             else:
                                 with torch.no_grad():
                                     outputs= model_inner(inputs)
                                     loss = criterion(outputs, labels)        
                             _, preds = torch.max(outputs,-1)
                             
                             count+=labels.shape[0]
                             labels_cpu = labels.data.cpu().numpy()                   
                             preds_cpu = preds.detach().cpu().numpy()
                             pred_save=np.append(pred_save, preds_cpu)
                             label_save=np.append(label_save, labels_cpu)
                            
                             running_loss += loss.item() 
                             running_corrects += np.sum(preds_cpu == labels_cpu)
                             tepoch.set_postfix(loss=running_loss/count, 
                                                        accuracy=100. * running_corrects/count)
                    f1=f1_score(label_save,pred_save,average=None)
                    epoch_loss = running_loss / count
                    epoch_acc = running_corrects /count
                    temp.append([epoch_acc,epoch_loss,f1[1]])
                    cm=confusion_matrix(label_save,pred_save,labels=[0,1])

                    if  phase == 'val':
                        list_l=temp[0]+temp[1]
                        val_acc_history.append(list_l)
                        temp=[]
                        df = pd.DataFrame(val_acc_history,columns=['train_acc', 'train_loss', 'train_f1', 'val_acc', 'val_loss', 'val_f1'])
                        dict_my=dict(zip(['train_acc', 'train_loss', 'train_f1', 'val_acc', 'val_loss', 'val_f1'],list_l))
                        writer.add_scalars('reports',dict_my, epoch)
                        df.to_excel(excel_epoch, sheet_name='fold{}'.format(k))
                     
                        if epoch_acc > best_loss:
                            best_loss=epoch_acc
                            torch.save(model.state_dict(), save_root_path+'/fold :{}_best'.format(k)+'.pth')
                            df2 = pd.DataFrame(cm,columns=['0','1'])

                            df2.to_excel(excel_best, sheet_name='fold{}'.format(k))
                             
                        if epoch==num_epochs-1:
                             torch.save(model.state_dict(), save_root_path+'/fold :{}_last'.format(k)+'.pth')
                             df2 = pd.DataFrame(cm,columns=['0','1'])
                             df2.to_excel(excel_last, sheet_name='fold{}'.format(k))  
        excel_last.save()
        excel_epoch.save()
        excel_best.save()

        excel_last.close()
        excel_epoch.close()
        excel_best.close()
   
                     
