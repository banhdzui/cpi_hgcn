'''
Created on 17 Sep 2019

@author: danhbuithi
'''
import sys 
import random

import torch
import torch.nn as nn
from torch import optim
from torch_geometric.data import DataLoader as gDataLoader

from common.command_args import CommandArgs
from common.pytorchtools import EarlyStopping

from cpi.dataset import CPIGnnCnnDataset
from cpi.models import CPIGnnCnnModel

from cpi.data_preprocessing import dump_dictionary
from cpi.visualization import evaluate_classification_result
from cpi.model_learning import train_cpi_predictor, test_cpi_predictor

from cpi.protein_model import ProteinSetting
from common.protein_utils import AMINO_ACID_NUMBER
from cpi.compound_model import CompoundSetting
from convert_data import load_gnn_data_meta


def create_model_configuration(config, compound_vocab_size, max_protein_length, 
                                  output_dim=2, protein_emb_dim=100):
    
    hidden_dim = int(config.get_value('hidden_dim'))
    
    compound_config = CompoundSetting(compound_vocab_size,
                                      hidden_dim,
                                      output_dim)
    
    
    nkernels = int(config.get_value('nkernel'))
    dropout_value = float(config.get_value('dropout'))
    protein_config = ProteinSetting(AMINO_ACID_NUMBER, 
                                    max_protein_length,
                                    hidden_dim, 
                                    protein_emb_dim,
                                    nkernels,
                                    dropout_value)
    return compound_config, protein_config



def create_dataset(root, file_name, compound_config, protein_config):
    
    return CPIGnnCnnDataset(root, 
                           file_name,
                           compound_config.vocab_size, 
                           protein_config.seq_length,
                           )

if __name__ == "__main__":
    
    config = CommandArgs({
                          'train'   : ('', 'Path of training data file'),
                          'test'   : ('', 'Path of testing data file'),
                          'val'     : ('', 'Path of validation data file'),
                          'root'    : ('', 'Path of root folder'),
                          'batchsize'   : (64, 'Input batch size'),
                          'hidden_dim'   : (128, 'compound/protein hidden state size'),
                          'lr'      :   (1e-4, 'Learning rate'),
                          'decay_weight'    :(1e-5, 'Weight of decay'),
                          'nloop'   : (100, 'Number of training iterations'),
                          'output'  : ('', 'Path of output file where save learned model'),
                          'meta'   : ('', 'Path of meta data file'),
                          'nkernel': (12, 'number of kernels for conv1d'),
                          'dropout': (0.2, 'dropout value'),
                          'model' : ('', 'Path of pre-trained model')
                          })    
    
    if not config.load(sys.argv):
        print ('Argument is not correct. Please try again')
        sys.exit(2)
        
    manual_seed = random.randint(1, 10000)
    torch.manual_seed(manual_seed)
    
    print('load arguments...')
    batch_size = int(config.get_value('batchsize'))
    weight_decay = float(config.get_value('decay_weight'))
    niter = int(config.get_value('nloop'))

    root = config.get_value('root')
    lr = float(config.get_value('lr'))
    print('lr ', lr, 'weight decay ', weight_decay)
    out_dim = 2
    
    print('loading meta file...')
    atom_dict, _, max_protein_length = load_gnn_data_meta(config.get_value('meta'))
    n_atoms = atom_dict.size() + 1 #one for unknown symbol
    max_protein_length += 2 # +2 for start and end symbol
    compound_config, protein_config = create_model_configuration(config, 
                                                                    n_atoms, 
                                                                    max_protein_length, 
                                                                    output_dim=out_dim)
    
    if (config.get_value('output')):
        dump_dictionary({'compound': compound_config,
                         'protein': protein_config}, config.get_value('output')+'.conf')
        
    print('dropout', protein_config.drop_out)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('used device ', device)
    
    
    print('Creating model....')
    net = CPIGnnCnnModel(compound_config, protein_config).to(device)
    net = net.double()
    
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    

    print('Loading training data ....')
    train_dataset = create_dataset(root, config.get_value('train'),
                                            compound_config, protein_config)
    train_dataloader = gDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataloader = None 
    training_stopper = None 
    if config.get_value('val'):
        print('Loading validation data ....')
        training_stopper = EarlyStopping(patience=10, verbose=True, delta=0, path=config.get_value('output'))
        val_dataset = create_dataset(root, config.get_value('val'),
                                              compound_config, protein_config)
        val_dataloader = gDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print('Start training classifier...')  
    criterion = nn.CrossEntropyLoss()
        
    if config.get_value('model'):
        net.load_state_dict(torch.load(config.get_value('model'), map_location=device))
        if val_dataloader is not None:
            _, _, _, val_loss = test_cpi_predictor(val_dataloader, net, device, criterion)
            training_stopper(val_loss, net)
        
    
    for epoch in range(niter):
        train_cpi_predictor(epoch, train_dataloader, net, criterion, optimizer, device, niter)
     
        if val_dataloader is not None:
            _, _, _, val_loss = test_cpi_predictor(val_dataloader, net, device, criterion)
            training_stopper(val_loss, net)
            
        
        if training_stopper is not None and training_stopper.early_stop:
            break

                
    print('Loading testing data ....')
    test_dataset = create_dataset(root, config.get_value('test'),
                                           compound_config, protein_config)
    test_dataloader = gDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    if val_dataloader is not None: 
        net.load_state_dict(torch.load(config.get_value('output'), map_location=device))
    elif config.get_value('output'): 
        torch.save(net.state_dict(), config.get_value('output'))

    print('Testing classifier...')  
    y_preds, y_pred_scores, y_trues, test_loss = test_cpi_predictor(test_dataloader, net, device, criterion)
    print('testing loss ', test_loss)
    evaluate_classification_result(y_trues, y_preds, y_pred_scores, draw_curve=False)

