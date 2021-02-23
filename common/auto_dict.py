'''
Created on 28 Mar 2020

@author: danhbuithi
'''

class AutoDict(object):
    '''
    classdocs
    '''


    def __init__(self, init_dict=None):
        '''
        Constructor
        '''
        self._dict = init_dict 
        if (self._dict is None):
            self._dict = {}
            
            
    def find(self, key):
        if key not in self._dict:
            n = len(self._dict)
            self._dict[key] = n
            #print('add new id ', n)
            
        return self._dict[key]
    
    
    def size(self):
        return len(self._dict)