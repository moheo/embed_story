import numpy as np
import torch


def mixIdx_withEpoch(epoch, len_in_seq, pad_len, mode='pre', givenFW=2, givenBW=2, verbose=False, **kwarg):
    N = len(len_in_seq)
    list_out_2_input = []
    len_in_seq = len_in_seq.cpu().numpy()
    
    for inst in range(N):
        one_out_2_input= []
        mixrate=sampling(epoch, **kwarg)
        if np.random.random_sample() < mixrate:
            if mode is 'pre':
                for i in np.arange(pad_len):
                    if pad_len-len_in_seq[inst]+givenFW <= i < pad_len-givenBW:
                        one_out_2_input.append( 1 )
                    else:
                        one_out_2_input.append( 0 )
            elif mode is 'post':
                for i in np.arange(pad_len):
                    if givenFW <= i < pad_len-len_in_seq[inst]-givenBW:
                        one_out_2_input.append( 1 )
                    else:
                        one_out_2_input.append( 0 )
            else:
                raise ValueError
        else:
            for i in np.arange(pad_len):
                one_out_2_input.append( 0 )
        list_out_2_input.append( one_out_2_input )           
    
    #print( 'Number of seq:', len(list_out_2_input))
    #print( 'lens: ', [len(list_out_2_input[i]) for i in range(N)] )
    #print( 'weird case: ', [list_out_2_input[i] for i in range(N) if len(list_out_2_input[i])==1] )
    if verbose:
        print( 'Epoch: ', epoch, ', Mix Rate: ', mixrate)
    return torch.from_numpy( np.array(list_out_2_input) ), mixrate

def buildMask(len_in_seq, pad_len, givenSet, mode='pre', givenFW=2, givenBW=2, **kwarg):
    #givenSets = [('ff',5,2),('ff',10,2),('ff',15,2),('ff',20,2),('ff',25,2),
    #             ('md',5,0),('md',10,0),('md',15,0),('md',20,0),('md',25,0),
    #             ('bw',2,5),('bw',2,10),('bw',2,15),('bw',2,20),('bw',2,25)]
    assert type(givenSet) == 'tuple', 'Not proper setting input'
    t_type, outGivenFW, outGivenBW = givenSet # ('ff',15,2)
    N = len(len_in_seq)
    list_out_2_input = []
    len_in_seq = len_in_seq.cpu().numpy()
    
    for inst in range(N):
        one_out_2_input= []
        one_len = len_in_seq(inst)
        givenFW, given_BW = getBorderMargin(one_len, t_type, outGivenFW, outGivenBW, givenFW=givenFW, givenBW=givenBW )
        if mode is 'pre':
            for i in np.arange(pad_len):
                if pad_len-len_in_seq[inst]+givenFW <= i < pad_len-givenBW:
                    one_out_2_input.append( 1 )
                else:
                    one_out_2_input.append( 0 )
        elif mode is 'post':
            for i in np.arange(pad_len):
                if givenFW <= i < pad_len-len_in_seq[inst]-givenBW:
                    one_out_2_input.append( 1 )
                else:
                    one_out_2_input.append( 0 )
        else:
            raise ValueError
            
        list_out_2_input.append( one_out_2_input )           
    
    return torch.from_numpy( np.array(list_out_2_input) )


def getBorderMargin( L, t_type, outGivenFW, outGivenBW, givenFW=2, givenBW=2 ):
    if outGivenFW < L-2 and outGivenBW < L-2:
        if t_type=='ff':
            givenFW=L-outGivenFW
            givenBW=outGivenBW
        elif t_type=='bw':
            givenFW=outGivenFW
            givenBW=L-outGivenBW
        else:
            outGivenSum=outGivenFW
            if (L+outGivenSum) % 2 == 1:
                givenFW=L-(L+outGivenSum)/2-1 #L-outGivenFW
                givenBW=L-(L+outGivenSum)/2 
            else:
                givenFW=L-(L+outGivenSum)/2 #L-outGivenFW
                givenBW=L-(L+outGivenSum)/2 

    return givenFW, givenBW

'''

def mixValue_withEpoch(X, pred, i, **kwarg):
    #print('(getCombined) X.shape:',X.shape ) # (96, 1, 52)
    #print('(getCombined) np.array(pred).shape:',np.array(pred).shape )  # (96, 1, 52)
    pred_in = np.squeeze( np.array(pred), axis=(1,) ) # (Y) 2 to t=T-1
    #print('(getCombined) pred_in.shape:',pred_in.shape )
    #print('(getCombined) X.shape:',X.shape )
    
    if np.random.random_sample() < sampling(i, **kwarg):
        newInp = X                                  #  (FF) 1 to t=T-2 , (BW) t=T to t=3
    else:
        newInp = np.vstack( (X[0,:], pred_in[:-1,:]) )   # (pred_FF) 2 to t=T-1, (pred_BW) T-1 to t=2
 
    #print('(getCombined) newInp.shape:',newInp.shape ) 
    return np.expand_dims(newInp, axis=0)
'''

def sampling(i, epsilon=0.01, k=0.99, c=0.001, kk=300.):
    #epsilon=0.01 # minimum amount of truth 
    #k=0.99   # initial value
    #c=0.001 # slope of the decay. (if c=0.001, it will be 0 at 500 epoch)

    #kk=300.    
    ## sum
    #newInp = X + pred_in
    
    # scheduled sampling
    # S. Bengio et al., Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks, NIPS 2015. 
    # https://arxiv.org/abs/1506.03099

    ## linear decay (eps_i = max(epsilon,k-ci)),    0 <= epsilon < 1
    #eps_i=max(epsilon, k - c * i)
    # inverse sigmoid decay (eps_i = k / (k + exp(i/k))),   k >= 1 (time index prob 0.5)
    eps_i=kk/(kk+np.exp(i/kk))
    return eps_i    