# loading and preprocessing "pororo data"
from __future__ import print_function
import numpy as np
# fix random seed for reproducibility
np.random.seed(1337)



# almost every server has the same configuration as follows:
# =================================
datapath='/root/data/data_pororo/'
savepath='/root/data/save_pororo/'
# =================================

def loadEmbeddedSKVec():
    # diagEpisodes: (171,) --- (35,4800), ...
    #inp = np.load(datapath + 'diagEpisodes.npy')
    #inp = np.load(datapath + 'pororo_combVec_rebuild.npy') # (16066,9600)
    inp = np.load(datapath + 'combEmbEpisodes_rebuild.npy') # (171,) --- (35,50), ...
    return inp

def loadSKVec():
    inp = np.load(datapath + 'pororo_combVec_rebuild.npy') # (16066,9600)
    stendIdx = np.load(datapath + 'stendIdx.npy') # (171, 2)
    inp_re = []
    for st,en in stendIdx:
        inp_re.append(inp[st:en,:])
    return inp_re

def getDataSplit(inp, test_idx, valid_idx):
    X_all=inp

    test_indices = [i for i in range(len(X_all)) if i % 10 in test_idx]
    valid_indices = [i for i in range(len(X_all)) if i % 10 in valid_idx]
    train_indices = [i for i in range(len(X_all)) if i not in test_indices + valid_indices]

    print( 'Splitting Pororo dataset => test: ',len(test_indices), ' train: ', len(train_indices), 
          'valid: ', len(valid_indices), ' sum:', len(test_indices)+len(train_indices)  )    

    X_train = [X_all[i] for i in train_indices]
    X_test = [X_all[i] for i in test_indices]
    X_valid = [X_all[i] for i in valid_indices]
    
    return X_train, X_test, X_valid


def pairingData(X_train, X_test, X_valid):
    pairSkVec_train = [zip(oneEp[:-1], oneEp[1:]) for oneEp in X_train]
    pairSkVec_test = [zip(oneEp[:-1], oneEp[1:]) for oneEp in X_test]
    pairSkVec_valid = [zip(oneEp[:-1], oneEp[1:]) for oneEp in X_valid]

    #i,t=0,0
    #print(len(pairSkVec_train)) #: 154
    #print(len(pairSkVec_train[i])) #: 35 (varible-length) (original length: 36)
    #print(len(pairSkVec_train[i][t])) #: 2
    #print(len(pairSkVec_train[i][t][0])) #: 50    
    return pairSkVec_train, pairSkVec_test, pairSkVec_valid


def preparePororo(inp_type='embed', valid_idx=[], test_idx=[9], direction='fw'):
    if inp_type=='embed':
        inp = loadEmbeddedSKVec()
    elif inp_type=='raw':
        inp = loadSKVec()
    else:
        raise ValueError
    
    if direction == 'bw':
        inp = [oneEp[::-1,:] for oneEp in inp]
        
    X_train, X_test, X_valid = getDataSplit(inp, test_idx, valid_idx)
    pairSkVec_train, pairSkVec_test, pairSkVec_valid = pairingData(X_train, X_test, X_valid)
    
    return X_train, X_test, X_valid, pairSkVec_train, pairSkVec_test, pairSkVec_valid

def getSavepath():
    return savepath