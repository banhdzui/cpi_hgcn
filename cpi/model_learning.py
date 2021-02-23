'''
Created on 23 Jun 2020

@author: danhbuithi
'''
import torch
import numpy as np 

from cpi.utils import import_classifier_result
from cpi.visualization import draw_heatmap
import cv2


def data_2_device(data, device):
    '''
    Convert data into tensors for GGNN-CNN1D model
    PARAMS:
    - data: a batch of data. It includes: Adjacent matrices, compounds features, protein features, label and indices in dataset.
    - device: cpu or gpu
    RETURNS:
    - Corresponding tensors
    '''
    g, gg, index = data 
    g.to(device)
    gg.to(device)
    return g, gg, index        
        

            
def train_cpi_predictor(epoch, data_loader, net, criterion, optimizer, device, niter):
    '''
    Train GGNN-CNN model for CPI problem
    PARAMS:
    - epoch: integer. Index of the epoch
    - data_loader: CPIDataLoader
    - net: pytorch model.
    - criterion: loss function 
    - optimizer: Optimize method
    - device: cpu or gpu 
    - niter: number of epoches
    RETURN:
    None
    '''           
    train_losses = []
    net.train()
    for i, data in enumerate(data_loader, 0):
        
        net.zero_grad()
        g, gg, _ = data_2_device(data, device)    
    
        output = net(g, gg)
        
        target = g.y.to(device)
        
        loss = criterion(output, target)  
        train_losses.append(loss.item())
        loss.backward()
        
        optimizer.step()
          
        #if i % int(len(data_loader) / 10 + 1) == 0:# and opt.verbal:
        print('[%d/%d][%d/%d] Loss: %.4f' % (epoch, niter, i, len(data_loader), loss.data))
             
    print('avg loss: ', np.average(train_losses))
                       
def test_cpi_predictor( data_loader, net, device, criterion=None, threshold=0.5):
    '''
    Test GGNN-CNN model with data
    PARAMS:
    - data_loader: CPIDataLoader
    - net: pytorch model.
    - device: cpu or gpu 
    - criterion: loss function. Default is None, none of loss values is calculated. 
    - draw_curve: boolean. Draw ROC curve or not. Default is False
    - threshold: float. Threshold to predict a positive label. Default is 0.5
    RETURN:
    None
    '''    
    y_preds = []
    y_pred_scores = []
    y_trues = []
    
    test_losses = []
    net.eval()
    with torch.no_grad():
        for _, data in enumerate(data_loader, 0):
            
            g, gg, _ = data_2_device(data, device)
    
            output = net(g, gg)
            
            if criterion is not None:
                target = g.y.to(device)
                local_loss = criterion(output, target)
                test_losses.append(local_loss.item())
                #print('Test set: Average loss: {:.4f}'.format(local_loss))
            
            import_classifier_result(output, g.y, device, y_preds, y_pred_scores, y_trues, threshold)
         
    avg_loss = 0 
    if len(test_losses) > 0: avg_loss = np.average(test_losses)
    return y_preds, y_pred_scores, y_trues, avg_loss   


def compute_pooled_gradients(gradients): 
    x = gradients.cpu().numpy() #m x D x L
    a = np.mean(x, axis=-1) #m x D
    
    a = np.expand_dims(a, axis=2) #m x D x 1
    return a 
   
    

def explain_cpi_prediction(data_loader, net, device, max_length = None, file_name=None):
    net.eval()
    dataset = data_loader.dataset
    
    for _, data in enumerate(data_loader, 0):
        net.zero_grad()
        
        g, gg, indices = data_2_device(data, device)
        output = net(g, gg)
        
        '''
        Compute feature map 
        '''
        pred = output.argmax(dim=1)
        one_hot = torch.eye(2) 
        one_hot = one_hot[pred]
        one_hot = torch.sum(one_hot*output)
        
        one_hot.backward()
        
        gradients = net.protein_model.get_activation_gradient()
        pooled_gradients = compute_pooled_gradients(gradients) #m x D x 1
        
        activations = net.protein_model.get_activations(g.target).detach()
        activations = activations.cpu().numpy() #m x D x L
        activations = activations*pooled_gradients
        
        heatmap_values = np.sum(activations, axis=1)
        heatmap_values = np.maximum(heatmap_values, 0)
        
        
        '''
        normalize
        '''
        c = np.max(heatmap_values)
        if c > 0:
            heatmap_values /=  c#m x L

        '''
        Upsampling 
        '''
        
        L = net.protein_model.model_setting.seq_length
        m, _ = heatmap_values.shape
        for i in range(m): 
            print(g.y[i])
            y = heatmap_values[i]
            y = np.reshape(y, (1, -1))
            y = cv2.resize(y, (L, 1), interpolation=cv2.INTER_LINEAR)
            y = y.flatten()
            
            _, protein = dataset.get_smile_protein(indices[i])
            draw_heatmap(protein, y, max_length=max_length, output=file_name)
            
        