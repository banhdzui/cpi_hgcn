'''
Created on 17 Sep 2019

@author: danhbuithi
'''
import sys 
import random

import torch
import torch.nn as nn
from torch_geometric.data import DataLoader as gDataLoader

from common.command_args import CommandArgs

from cpi.data_preprocessing import load_pickle
from cpi.visualization import evaluate_classification_result
from cpi.models import CPIGnnCnnModel
from train_gnncnn_cpi_model import create_dataset
from cpi.model_learning import test_cpi_predictor


if __name__ == "__main__":
    
    config = CommandArgs({
                          'test'   : ('', 'Path of testing data file'),
                          'root'    : ('', 'Path of root folder'),
                          'batchsize'   : (64, 'Input batch size'),
                          'model' : ('', 'Path of pre-trained model')
                          })    
    
    if not config.load(sys.argv):
        print ('Argument is not correct. Please try again')
        sys.exit(2)
        
    manual_seed = random.randint(1, 10000)
    torch.manual_seed(manual_seed)
    
    print('load arguments...')
    batch_size = int(config.get_value('batchsize'))
    
    root = config.get_value('root')
    out_dim = 2
    
    config_dict = load_pickle(config.get_value('model')+'.conf')
    compound_config = config_dict['compound']
    protein_config = config_dict['protein']
        
    print(protein_config.vocab_size)
        
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('used device ', device)
    
    
    print('Creating model....')
    net = CPIGnnCnnModel(compound_config, protein_config).to(device)
    net = net.double()
    net.load_state_dict(torch.load(config.get_value('model'), map_location=device))
    
    print('Start training classifier...')  
    criterion = nn.CrossEntropyLoss()
        
    print('Loading testing data ....')
    test_dataset = create_dataset(root,config.get_value('test'),
                                  compound_config, protein_config)
    test_dataloader = gDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    
    print('Testing classifier...')  
    y_preds, y_pred_scores, y_trues, test_loss = test_cpi_predictor(test_dataloader, net, device, criterion)
    print('testing loss ', test_loss)
    evaluate_classification_result(y_trues, y_preds, y_pred_scores, draw_curve=False)

