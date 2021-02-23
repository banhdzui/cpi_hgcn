'''
Created on 21 Oct 2019

@author: danhbuithi
'''
import h5py
import json
import pickle

import numpy as np
import networkx as nx
from rdkit import Chem

from common.auto_dict import AutoDict

def get_atom_ids(mol):
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = atoms[i]+'@' # change atom in aromatic to atom*
    
    return atoms 

def get_bond_type(bond):
    bond_type = bond.GetBondType()
    x = str(int(bond_type))
    if bond.GetIsAromatic() == True:
        x = x + '@'
    return x 
    
def load_gnn_hdf5_file(file_name):
    compound_protein_list = []
    with h5py.File(file_name, 'r') as db:
        for i_data in db['data']:
            x = json.loads(i_data)
            _, _, _, _, smiles, protein, target = x 
            compound_protein_list.append((smiles, protein, target))
     
    return compound_protein_list

def split_aromatic_and_non_aromatic_subgraphs(mol):  
    g1 = nx.Graph()
    g2 = nx.Graph()
     
    if len(mol.GetAtoms()) == 1:
        g1.add_node(0)
    else:
        for bond in mol.GetBonds():
            a = bond.GetBeginAtomIdx()
            b = bond.GetEndAtomIdx()
            bond_type = get_bond_type(bond)
        
            if bond.GetIsAromatic() == True:
                g2.add_edge(a, b, label=bond_type)
            else: 
                g1.add_edge(a, b, label=bond_type)

    return g1, g2

def get_graph_information(g, node_id_list, node_start_id = 0, comp_start_id = 0):
    new_node_ids = AutoDict()
    [new_node_ids.find(x) for x in g.nodes]
    
    atoms = [node_id_list[c] for c in g.nodes]
    
    edges = []
    for u, v in g.edges:
        x = new_node_ids.find(u) + node_start_id
        y = new_node_ids.find(v) + node_start_id
        edges.append((x, y))
        edges.append((y, x))
    
    cluster_id = [comp_start_id for _ in range(len(g.nodes))]
    
    components = sorted(nx.connected_components(g), key = len, reverse=True)
    ncomponents = len(components)
    
    for i in range(ncomponents):
        for j in list(components[i]):
            k = new_node_ids.find(j)
            cluster_id[k] += i
    return atoms, edges, cluster_id, components

def add_virtual_edges(components):
    n = len(components)
    edges = []
    for i in range(n-1):
        for j in range(i+1, n):
            if components[i].intersection(components[j]):
                edges.append((i, j))
                edges.append((j, i))
    return edges 
        
def convert_2_gnn_data(compound_protein_list, node_dict = None):
    if node_dict is None:
        node_dict = AutoDict()
        
    transformed_data = []
    max_nvertices = 0
    for smiles, sequence, interaction in compound_protein_list:
        #try:
        mol = Chem.MolFromSmiles(smiles)
        g1, g2 = split_aromatic_and_non_aromatic_subgraphs(mol)
    
        atom_symbols = get_atom_ids(mol)
        node_id_list = [node_dict.find(x) for x in atom_symbols]
    
    
        atoms1, edges1, cluster_id_1, components1 = get_graph_information(g1, node_id_list)
        atoms2, edges2, cluster_id_2, components2 = get_graph_information(g2, node_id_list, len(atoms1), len(components1))
        
        global_edges = add_virtual_edges(components1 + components2)
        
        nvertices = len(node_id_list)
    
        if max_nvertices < nvertices: max_nvertices = nvertices
    
        '''
        Update new indices
        '''
        transformed_data.append((atoms1+atoms2, edges1 + edges2, cluster_id_1 + cluster_id_2, global_edges, smiles, sequence, interaction))
    
        #except:
        #    print('error while parsing molecule ', smiles)
    return transformed_data, node_dict, max_nvertices

'''
Find word dictionary, the length and number of words of the longest protein.
'''
def find_length_of_longest_sample(compound_protein_list):
    protein_length = 0
    smiles_length = 0
    
    for smiles, protein_sequence, _ in compound_protein_list:
        '''
        Update the size of protein
        '''
        if protein_length < len(protein_sequence):
            protein_length = len(protein_sequence)
        if smiles_length < len(smiles):
            smiles_length = len(smiles)
    return protein_length, smiles_length


'''
Load data file in which each line contains smiles,protein sequence and interaction information
'''
def load_raw_data(file_name, remove_dot=True, int_label=True):
    compound_protein_list = []

    with open(file_name, 'r') as file_reader:
        for line in file_reader:
            
            smiles, sequence, interaction = line.strip().split()
            
            """Exclude data contains '.' in the SMILES format."""
            if remove_dot and '.' in smiles: continue
            if int_label:
                compound_protein_list.append((smiles, sequence, int(interaction)))
            else:
                compound_protein_list.append((smiles, sequence, float(interaction)))
    return compound_protein_list
            
            
'''
Split data into k folds. Return the indices of data points in each fold
'''
def split_data_in_folds(n, k=5):
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    p = int(n/k)
    subsets = []
    start = 0
    for _ in range(k-1):
        end = start + p 
        subsets.append(indices[start:end])
        start = end 
    subsets.append(indices[end:])
    return subsets 
    
'''
Split data into 2 parts. Return the indices of data points in each part
'''
def split_data_in_two(n, training_rate):
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    k = int(training_rate * n)
    return indices[:k], indices[k:]
 
def copy_data_by_indices(data_set, indices):
    copied_set = []
    for i in indices:
        if i >= len(data_set): continue
        copied_set.append(data_set[i])
    return copied_set 

 
def save_data_in_hdf5_format(file_name, data_set):
    file_writer = h5py.File(file_name, 'w')
    dt = h5py.string_dtype(encoding='ascii')
    dset = file_writer.create_dataset('data', (len(data_set),), dtype=dt, compression='gzip')
    for i, x in enumerate(data_set):
        dset[i] = json.dumps(x)
    file_writer.close()
    
    
def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dict(dictionary), f)