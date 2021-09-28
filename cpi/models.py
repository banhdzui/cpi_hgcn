'''
Created on 28 Apr 2020

@author: danhbuithi
'''

import torch
import torch.nn as nn

from cpi.protein_model import CNNProteinEncoder
from cpi.compound_model import GNNCompoundEncoder


        
class CPIModel(nn.Module):
    '''
    classdocs
    '''
    def __init__(self, compound_config, protein_config):
        '''
        Constructor
        '''
        super(CPIModel, self).__init__()
        self.compound_config = compound_config
        self.protein_config = protein_config
    
        '''
        compound model
        '''
        self.compound_model = self.create_compound_model()
          
        '''
        protein model
        '''
        self.protein_model = self.create_protein_model()
        
        self.output_layer = self.create_output_layer()
        
        print('use global version')
             
    def create_protein_model(self):
        return CNNProteinEncoder(self.protein_config)
            
    def create_compound_model(self):
        return None
        
        
    def create_output_layer(self):
        out_dim = self.compound_config.output_dim
        hidden_dim = self.compound_config.hidden_dim
        
        return nn.Sequential(nn.Linear(2 * hidden_dim, 512),
                                               nn.ReLU(),
                                               nn.Dropout(0.1),
                                               nn.Linear(512, 512),
                                               nn.ReLU(),
                                               nn.Dropout(0.1),
                                               nn.Linear(512, 64),
                                               nn.ReLU(),
                                               nn.Linear(64, out_dim)
            )
                   
        
    '''
    Forward state
    - prop_state: m x V x D
    - annotation: m x V x d
    - protein_x: m x seq_len x 1 x D'
    - protein_h: nlayer x m x hidden_size
    '''
    def forward(self, compound_state, protein_state):
        
        '''
        Forward step for Compounds
        '''
        cx = self.compound_model(compound_state) #m x D
        '''
        Forward step for Proteins
        '''
        px = self.protein_model(protein_state) #m x D
        
        '''
        Join compounds and proteins
        '''
        joint_state = torch.cat((cx,px), dim=-1) #m x 2*D
        
        
        return self.output_layer(joint_state)
    
                
class CPIGnnCnnModel(CPIModel):
    '''
    classdocs
    '''

    def __init__(self, compound_config, protein_config):
        '''
        Constructor
        '''
        CPIModel.__init__(self, compound_config, protein_config)
        self.full_dropout = nn.Dropout(protein_config.drop_out)
                
    def create_protein_model(self):
        return CNNProteinEncoder(self.protein_config)
    
    def create_compound_model(self):
        return GNNCompoundEncoder(self.compound_config)
        
    
    '''
    Forward state
    - prop_state: m x V x D
    - annotation: m x V x d
    - protein_x: m x seq_len x 1 x D'
    - protein_h: nlayer x m x hidden_size
    '''
    def forward(self, g, gg):
        '''
        Forward step for Compounds
        '''
        cx = self.compound_model(g, gg) #m x D
        '''
        Forward step for Proteins
        '''
        px = self.protein_model(g.target) #m x D
        
        '''
        Join compounds and proteins
        '''
        joint_state = torch.cat((cx,px), dim=-1) #m x 2D
        joint_state = self.full_dropout(joint_state)
        
        return self.output_layer(joint_state)
    