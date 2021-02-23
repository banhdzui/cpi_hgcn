'''
Created on 17 Sep 2019

@author: danhbuithi
'''

import json
import h5py
import numpy as np
import pandas as pd 

import torch 

from torch_geometric.data import Dataset as gDataset
from torch_geometric import data as gDATA
from common.protein_utils import ProteinUtils, BEGIN_SYMBOL,\
    END_SYMBOL

class CPIGnnCnnDataset(gDataset):
    def __init__(self, root, file_name, atom_vocab_size, max_protein_length): 
        super(CPIGnnCnnDataset, self).__init__(root)
        self.atom_vocab_size = atom_vocab_size
        
        self.max_protein_length = max_protein_length
        
        self.hdf5file = file_name
        with h5py.File(self.hdf5file, 'r') as db:
            self.nsamples =len(db['data'])
            
    
    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass 

    def download(self):
        pass
    
    def _download(self):
        pass 
        
    def len(self):
        return self.nsamples 
    
    def get_smile_protein(self, index):
        with h5py.File(self.hdf5file, 'r') as db:
            i_data = json.loads(db['data'][index])
            
        return i_data[4], i_data[5]
    
    def _build_compound_graph_data(self, atoms, edges, target = None):
        atoms = np.reshape(atoms, (-1, 1)) #m x 1
        atoms[atoms >= self.atom_vocab_size] = self.atom_vocab_size-1
        
        edges = np.array(edges)
        if edges.shape[0] != 0:
            edges = np.transpose(edges)
        edges = torch.LongTensor(edges)
            
        if target is not None: 
            return gDATA.Data(x=torch.LongTensor(atoms),
                                edge_index= edges,
                                y = torch.LongTensor([target])
                                )
            
        return gDATA.Data(x=torch.LongTensor(atoms),
                                edge_index= edges
                                )
            
    def _get_protein_features(self, protein):
        protein = BEGIN_SYMBOL + protein + END_SYMBOL
        n = min(len(protein), self.max_protein_length) #end and begin sybmol
        protein = protein[:n]
        
        aa_indices = np.zeros(self.max_protein_length) #used for padding
        P = np.array([ProteinUtils.amino_acid_index(c) for c in protein])
        aa_indices[:n] = P[:n]
        return aa_indices 
        
    def _extract_feature(self, data):
        
        atoms, edges, cluster_index, global_edges, _, protein, target = data
        
        g = self._build_compound_graph_data(atoms, edges, target)
        g.cluster_index = torch.LongTensor(cluster_index)
        
        n = max(cluster_index) + 1
        gg = self._build_compound_graph_data(np.arange(n), global_edges)
        
        aa_indices = self._get_protein_features(protein)
        g.target = torch.LongTensor([aa_indices])
        
        return g, gg
    
    def get(self, idx):
        with h5py.File(self.hdf5file, 'r') as db:
            i_data = json.loads(db['data'][idx])
        
        g, gg = self._extract_feature(i_data) 
        return g, gg, idx
    
    def get_labels(self):
        result = []
        with h5py.File(self.hdf5file, 'r') as db:
            for x in db['data']: 
                i_data = json.loads(x)
                result.append(i_data[-1])
        return result 
        
                
    def save_prediction_in_csv(self, y_preds, y_pred_scores, output_file):
        output_result = []
        with h5py.File(self.hdf5file, 'r') as db:
            i = 0
            for x in db['data']: 
                i_data = json.loads(x)
                _, _, _, _, smiles, protein, target = i_data 
                output_result.append((smiles, protein, target, y_preds[i], y_pred_scores[i]))
                i += 1
    
        df = pd.DataFrame(output_result, columns =['compound', 'protein', 'true label','class', 'score'])
        df.to_csv(output_file, index=False)
  

        