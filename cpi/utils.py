'''
Created on 23 Jun 2020

@author: danhbuithi
'''
import torch
import numpy as np 
import torch.nn.functional as F 

def create_len_mask(max_len, xlens):
    a = torch.arange(max_len, device=xlens.device)[None,:] < xlens[:,None]
    return a
    
def import_classifier_result(output, target, device, y_preds, y_pred_scores, y_trues, threshold):
    pred = F.softmax(output, 1).to(device)
    predicted_scores = pred[:,1].tolist()
    predicted_labels = (np.array(predicted_scores) > threshold).astype(int)
    
    y_preds.extend(predicted_labels)
    y_pred_scores.extend(predicted_scores)
    y_trues.extend(target.tolist())
    
    
def import_regression_result(output, target, y_pred_scores, y_trues):
    predicted_scores = output.tolist()
    y_pred_scores.extend(predicted_scores)
    true_labels = target.tolist()
    y_trues.extend(true_labels)