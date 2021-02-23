'''
Created on 07 Jan 2020

@author: danhbuithi
'''

import math 
import torch 
import torch.nn as nn


class ProteinSetting(object):
    '''
    Configuration for a ProteinGNN model
    '''    
    def __init__(self, vocab_size, seq_length, output_dim, emb_dim, n_kernels=12, drop_out=0.2):
        '''
        Constructor
        PARAMS:
        - vocab_size: integer. Vocabulary size (#words)
        - seq_length: integer. The fixed length of protein sequence.
        - emb_dim: integer. Dimension of embedding layer
        - output_dim: integer. Dimension of output layer
        - n_kernels: integer. Number of kernels is used in CNN1D layers.
        '''
        self.vocab_size  = vocab_size
        self.seq_length = seq_length
        
        self.emb_dim = emb_dim 
        self.output_dim = output_dim 
        
        self.n_kernels = n_kernels
        self.drop_out = drop_out
        
        print('protein vocab size ', self.vocab_size)
        print('seq length ', self.seq_length)
        print('protein hidden dim ', self.output_dim)
        
def compute_reduced_size(L, used_conv_kernels):
    for c, m in used_conv_kernels:
        if c is not None: 
            L = L - c + 1 
        if m is not None:
            L = math.floor((L - m)/m + 1)
    return L 
    
    
class CNNProteinEncoder(nn.Module):
    '''
    A model representing protein sequence. The model consists of one embedding layer, 2 CNN-1D layers.
    '''

    def __init__(self, model_setting):
        '''
        constructor
        PARAMS:
        - model_setting: ProteinSetting. It contains necessary configuration of the model 
        '''
        super(CNNProteinEncoder, self).__init__()
        
        self.model_setting = model_setting
        
        nkernels = self.model_setting.n_kernels
        emb_dim = self.model_setting.emb_dim 
        self.embedding = nn.Embedding(self.model_setting.vocab_size, 
                                      emb_dim,
                                      padding_idx=0)
        
        start = 64
        step = 16
        self.conv_layers = nn.Sequential(nn.Conv1d(emb_dim, start, kernel_size=3),
                                         nn.ReLU(),
                                        nn.Conv1d(start, start+step, kernel_size=nkernels),
                                         nn.ReLU(),
                                         nn.MaxPool1d(kernel_size=nkernels),
                                        nn.Conv1d(start+step, start+step*2, kernel_size=nkernels),
                                        nn.ReLU()
                                        )
        
        self.average_pool = nn.MaxPool1d(kernel_size=12)
        
        used_kernels = [(3,None), (nkernels, nkernels), (nkernels, 12)]
        self.L = compute_reduced_size(self.model_setting.seq_length, used_kernels)
        print(self.L,self.L * (start + 2*step))
        out_dim = self.model_setting.output_dim
        self.dropout = nn.Dropout(self.model_setting.drop_out)
        self.output_layer = nn.Sequential(nn.Linear(self.L * (start + 2*step), 512),
                                           nn.ReLU(),
                                           nn.Linear(512, out_dim)
                                           )
        
        self.gradients = None 
                  
    def activation_hook(self, grad):
        self.gradients = grad 
        
    def get_activation_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2) # m x emb_dim x L1
        
        x = self.conv_layers(x) #m x D x L
        return x
        
        
    def forward(self, x):
    
        x = self.embedding(x) # m x L x emb_dim
        x = x.transpose(1, 2) # m x emb_dim x L1
        
        x = self.conv_layers(x) #m x D x L
        
        '''
        un-comment this when doing explanation
        '''
        #x.register_hook(self.activation_hook)
        
        x = self.average_pool(x)
        x = x.transpose(1, 2).contiguous() #m x L x D
        x = torch.flatten(x, start_dim = 1)
        x = self.dropout(x)
        return self.output_layer(x)
        
    