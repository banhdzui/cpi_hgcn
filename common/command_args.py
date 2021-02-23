'''
Created on May 3, 2017

@author: BanhDzui
'''
import getopt

class CommandArgs(object):
    
    '''
    In arguments dictionary, key is name of argument and value includes default value and explanation
    '''
    def __init__(self, args_dict):
        self.args_dict = args_dict

    def load(self, argv):
        try:
            args_list = [(key + '=') for key in self.args_dict.keys()]
            opts, _ = getopt.getopt(argv[1:], '', args_list)
        except getopt.GetoptError:
            for key, value in self.args_dict.items():
                print(key + ':' + value[1])
            return False
        for opt, arg in opts:
            opt_key = opt[2:]
            description = self.args_dict[opt_key][1]
            self.args_dict[opt_key] = (arg, description)
        return True    

    def get_value(self, arg_name):
        return self.args_dict[arg_name][0]
        