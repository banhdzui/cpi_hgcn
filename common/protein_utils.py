'''
Created on 28 Nov 2019

@author: danhbuithi
'''

AMINO_ACID_NUMBER = 25 #include unknown amino acid
AA_CATEGORIES_NUMBER = 9
BEGIN_SYMBOL = '>'
END_SYMBOL = '$'

class ProteinUtils(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
        
        
    @staticmethod 
    def amino_acids():
        return 'ACDEFGHIKLMNPQRSTUVWYO'
    
    
    @staticmethod
    def amino_acid_indices():
        aa_list = ProteinUtils.amino_acids()
        aa_dict = {}
        for i in range(len(aa_list)):
            aa_dict[aa_list[i]] = i 
        return aa_dict 
    
    @staticmethod 
    def amino_acid_index(c):
        if c == 'A': return 1 
        elif c == 'C': return 2 
        elif c == 'D': return 3 
        elif c == 'E': return 4
        elif c == 'F': return 5 
        elif c == 'G': return 6
        elif c == 'H': return 7 
        elif c == 'I': return 8
        elif c == 'K': return 9
        elif c == 'L': return 10 
        elif c == 'M': return 11
        elif c == 'N': return 12 
        elif c == 'P': return 13 
        elif c == 'Q': return 14
        elif c == 'R': return 15 
        elif c == 'S': return 16
        elif c == 'T': return 17 
        elif c == 'U': return 18
        elif c == 'V': return 19 
        elif c == 'W': return 20
        elif c == 'Y': return 21
        elif c == 'O': return 22
        elif c == BEGIN_SYMBOL: return 23 
        elif c == END_SYMBOL: return 24
        return 0
    
            