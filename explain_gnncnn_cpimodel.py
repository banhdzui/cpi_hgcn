'''
Created on 17 Sep 2019

@author: danhbuithi
'''
import sys 
import random
from common.command_args import CommandArgs

import torch
from torch_geometric.data import DataLoader as gDataLoader

from cpi.data_preprocessing import load_pickle
from cpi.models import CPIGnnCnnModel
from train_gnncnn_cpi_model import create_dataset
from cpi.model_learning import explain_cpi_prediction

if __name__ == "__main__":
    
    config = CommandArgs({
                          'test'   : ('', 'Path of testing data file'),
                          'root'    : ('', 'Path of root folder'),
                          'batchsize'   : (64, 'Input batch size'),
                          'model'   : ('', 'Path of existing model'),
                          'output'   : ('', 'Path of output file')
                          })    
    
    if not config.load(sys.argv):
        print ('Argument is not correct. Please try again')
        sys.exit(2)
        
    manual_seed = random.randint(1, 10000)
    torch.manual_seed(manual_seed)
    
    print('load arguments...')
    batch_size = int(config.get_value('batchsize'))
    root = config.get_value('root')
    print('batch size', batch_size)
    print('loading meta file...')
    
    config_dict = load_pickle(config.get_value('model') + '.conf')
    compound_config = config_dict['compound']
    protein_config = config_dict['protein']
    
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print('used device ', device)
    
    print('Creating model....')
    net = CPIGnnCnnModel(compound_config, protein_config).to(device)
    net.double()
    
    print('loading trained model')
    net.load_state_dict(torch.load(config.get_value('model'), map_location=device))
                    
    print('Loading testing data ....')
    test_dataset = create_dataset(root, config.get_value('test'),
                            compound_config, protein_config)
    test_dataloader = gDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    output_file = None
    if config.get_value('output'):
        output_file = config.get_value('output') 
    explain_cpi_prediction(test_dataloader, net, device, max_length=1000, file_name=output_file)
    
