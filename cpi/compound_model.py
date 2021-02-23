'''
Created on 16 May 2020

@author: danhbuithi
'''
import torch
from torch import nn
from torch import Tensor 
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.norm.batch_norm import BatchNorm
from torch_geometric.nn.glob.glob import global_max_pool#, global_mean_pool
from torch_geometric.nn.pool.max_pool import max_pool_x

class CompoundSetting(object):
    '''
    Configuration for compound model
    '''
    def __init__(self, vocab_size, hidden_dim, output_dim=2, drop_out=0.1):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim 
        self.output_dim = output_dim
        self.drop_out=drop_out
        print('compound vocab size ', self.vocab_size)
        print('compound hidden dim ', self.hidden_dim)
        print('output dim ', self.output_dim)

    
class GWLConv(MessagePassing):
    def __init__(self, atom_channel: int,
                 out_channel: int, aggr="add",
                 **kwargs):
        super(GWLConv, self).__init__(aggr, **kwargs)
            
        #self.compress_nn = nn.Linear(atom_channel, atom_channel)
        self.fc = nn.Linear(2*atom_channel, out_channel)
        
    def reset_parameters(self):
        #self.compress_nn.reset_parameters()
        self.fc.reset_parameters()
        
        
    def forward(self, x, edge_index, size=None):

        if isinstance(x, Tensor):
            x = (x, x)
            
        if edge_index.shape[-1] > 0: 
            out = self.propagate(edge_index, x = x, size=size)
        else:
            out = torch.zeros_like(x[0])
        
        
        x_r = x[1]
        if x_r is not None:
            out = torch.cat([out, x_r], dim=-1)
            out = F.relu(self.fc(out))
        #print(out)    
        return out 

            
class GNNCompoundEncoder(nn.Module):
    
    def __init__(self, compound_config):
        super(GNNCompoundEncoder, self).__init__()
        self.compound_config = compound_config 
        
        self.compound_config = compound_config 
        
        atom_dim = 64
        step = 16
        mol_dim = compound_config.hidden_dim 
        
        nembeddings = compound_config.vocab_size
        self.atom_embed_layer = nn.Embedding(nembeddings, atom_dim, padding_idx=nembeddings-1)
        
        k = 2
        self.atom_convs = nn.ModuleList([GWLConv(atom_dim+i*step, atom_dim+(i+1)*step) for i in range(k)])
        self.atom_bns = nn.ModuleList([BatchNorm(atom_dim+(i+1)*step) for i in range(k)])
        
        self.mol_conv = GWLConv(atom_dim+k*step, mol_dim)
        

    def forward(self, g, gg):
        '''
        Forward step for Compounds
        '''
        x, edge_index, batch, cluster_index = g.x, g.edge_index, g.batch, g.cluster_index
        
        x = torch.squeeze(x, -1)
        x = self.atom_embed_layer(x)
    
        
        for i, convi in enumerate(self.atom_convs):
            x = convi(x, edge_index)
            x = self.atom_bns[i](x)
            
        y, ybatch = max_pool_x(cluster_index, x, batch)
        
        global_edge_index = gg.edge_index
        y = self.mol_conv(y, global_edge_index)
        y = global_max_pool(y, ybatch)
        return y 
        