
import sys 

from common.command_args import CommandArgs
    
from cpi.data_preprocessing import load_raw_data, convert_2_gnn_data
from cpi.data_preprocessing import save_data_in_hdf5_format

from convert_data import load_gnn_data_meta
        
if __name__ == '__main__':
    config = CommandArgs({
                          'in'   : ('', 'Input file containing cpi dataset'),
                          'meta'   : ('', 'path of meta file '),
                          'out'   : ('', 'Output file containing CPI'),
                          'use_dot': ('n', 'data for classification')
                          })    
    
    if not config.load(sys.argv):
        print ('Argument is not correct. Please try again')
        sys.exit(2)
        
    option = config.get_value('option')    
    radius = int(config.get_value('radius'))
    
    remove_dot = True 
    if config.get_value('use_dot') == 'y': 
        remove_dot = False
        
    
    print('loading dataset...')
    compound_protein_list = load_raw_data(config.get_value('in'), remove_dot=remove_dot, int_label=True)
    
    print('Convert data into gnn format...')
    gnn_atom_dict, _, _ = load_gnn_data_meta(config.get_value('meta'))
    gnn_dataset, _, _ = convert_2_gnn_data(compound_protein_list, gnn_atom_dict)
    save_data_in_hdf5_format(config.get_value('out'), gnn_dataset)
