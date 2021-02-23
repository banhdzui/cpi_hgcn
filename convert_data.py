'''
Created on 21 Oct 2019

@author: danhbuithi
'''
import sys
import numpy as np 
from common.command_args import CommandArgs

from cpi.data_preprocessing import load_raw_data, convert_2_gnn_data
from cpi.data_preprocessing import find_length_of_longest_sample
from cpi.data_preprocessing import save_data_in_hdf5_format
from cpi.data_preprocessing import split_data_in_folds
from cpi.data_preprocessing import copy_data_by_indices
from cpi.data_preprocessing import load_pickle, dump_dictionary
    
    
def save_gnn_data_meta(file_name, atom_dict, max_nvertices, max_protein_length):
    gnn_dict = {'atom': atom_dict, 
                'nvertices': max_nvertices,
                'protein': max_protein_length
                }
    dump_dictionary(gnn_dict, file_name)
    
def load_gnn_data_meta(file_name):
    gnn_dict = load_pickle(file_name)
    return gnn_dict['atom'], gnn_dict['nvertices'], gnn_dict['protein']

def create_train_test_data(n, rate = 0.2):
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    ntest = int(rate*n)
    ntrain = n - ntest 
    train_indices = indices[:ntrain]
    test_indices = indices[ntrain:]
    return train_indices, test_indices

def create_train_val_test_data(n, rate=0.2):
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    ntest = int(rate*n)
    ntrain = n - 2*ntest 
    train_indices = indices[:ntrain]
    val_indices = indices[ntrain:(ntrain+ntest)]
    test_indices = indices[(ntrain+ntest):]
    return train_indices, val_indices, test_indices

def create_fold_data(n, k):
    index_subsets = split_data_in_folds(n, k)
    folds = []
    for i in range(k):
        test_indices = index_subsets[i]
        train_indices = []
        for j in range(k):
            if j == i: continue 
            train_indices.extend(index_subsets[j])
        folds.append((train_indices, test_indices))
    return folds

def generate_indices_4_splits(n, k, rate=0.2):
    if k <= 1: return None 
    if k == 2: return create_train_test_data(n, rate)
    if k == 3: return create_train_val_test_data(n, rate)
    return create_fold_data(n, k)     

def save_data_in_splits(data, folds, k, original_file):
    if k <= 1:
        save_data_in_hdf5_format(original_file + '.gnn.train', data)
    elif k == 2: 
        train_indices, test_indices = folds 
        my_train = copy_data_by_indices(data, train_indices)
        save_data_in_hdf5_format(original_file + '.gnn.train', my_train)
        
        my_test = copy_data_by_indices(data, test_indices)
        save_data_in_hdf5_format(original_file + '.gnn.test', my_test)
    elif k == 3: 
        train_indices, val_indices, test_indices = folds 
        my_train = copy_data_by_indices(data, train_indices)
        save_data_in_hdf5_format(original_file + '.gnn.train', my_train)
    
        my_val = copy_data_by_indices(data, val_indices)
        save_data_in_hdf5_format(original_file + '.gnn.val', my_val)
        
        my_test = copy_data_by_indices(data, test_indices)
        save_data_in_hdf5_format(original_file + '.gnn.test', my_test)
    else: 
        for i, fold in enumerate(folds):
            train_indices, test_indices = fold
            my_train = copy_data_by_indices(data, train_indices)
            save_data_in_hdf5_format(original_file + '.gnn.train.'  + str(i), my_train)
        
            my_test = copy_data_by_indices(data, test_indices)
            save_data_in_hdf5_format(original_file + '.gnn.test.' + str(i), my_test)
    
            
if __name__ == '__main__':
    config = CommandArgs({
                          'in'   : ('', 'Path of data file'),
                          'option': (0, 'Option for splitting data, an integer to denote number of splits'),
                          'rate'   : (0.2, 'Rate of testing data'),
                          'use_dot': ('n', 'data for classification')
                          })    
    
    if not config.load(sys.argv):
        print ('Argument is not correct. Please try again')
        sys.exit(2)
        
        
    option = int(config.get_value('option'))
    original_file = config.get_value('in')
    r = float(config.get_value('rate'))
    
    print('Loading dataset...')
    remove_dot = True 
    if config.get_value('use_dot') == 'y': 
        remove_dot = False
        
    compound_protein_list = load_raw_data(original_file, remove_dot=remove_dot, int_label=True)
        
    
    n = len(compound_protein_list)
    folds = generate_indices_4_splits(n, option, r)
    max_protein_length, max_smiles_length = find_length_of_longest_sample(compound_protein_list)
        
    print('Converting data into torch geometric format')
    gnn_dataset, gnn_atom_dict, max_nvertices = convert_2_gnn_data(compound_protein_list)

    print('Writing meta data...')
    save_gnn_data_meta(config.get_value('in')+'.gnn.meta', 
                             gnn_atom_dict,
                             max_nvertices, 
                             max_protein_length)
    
    print('Saving data ...')
    save_data_in_splits(gnn_dataset, folds, option, original_file)
    