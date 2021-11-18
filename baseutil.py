
import os
import sys
import os.path
from os import path
import argparse
import numpy as np
import pandas as pd
import torch.optim as optim
import random
import torch
import time
import logging
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

def load_data(fname, factor, target, allFactors, exVars, event, include_complete=False,val_fraction=None, icovariates=None):

    """ Load data set """
    data_in = pd.read_csv(fname,  encoding = "ISO-8859-1", engine='python')
    covariates = list(set(data_in.columns) - {target, event} - set(exVars) - set(allFactors))
    
    if(include_complete):
        print('Exclude complete cases!!!')
        data_in = data_in[data_in[event] == 1]
    
    if(icovariates!= None):
        covariates = icovariates
        
    print('covariates')
    
    
    print(covariates)
    
    data = {}
    data['x'] = data_in[covariates].to_numpy().astype(np.float64)
    ##############
    '''
    scaler = StandardScaler()
    scaler.fit(data['x'])
    data['x'] = scaler.transform(data['x'])
    '''
    
    ###############
    data['t'] = data_in[factor].to_numpy()
    data['y'] = data_in[target].to_numpy()
    data['e'] = data_in[event].to_numpy()
    
    if(val_fraction == None):
    
        data['t'] = data['t'].reshape(len(data['t']),1)
        data['y'] = data['y'].reshape(len(data['y']),1)
        data['e'] = data['e'].reshape(len(data['e']),1) 
        
        data['x'] = torch.from_numpy(data['x'])
        data['t'] = torch.from_numpy(data['t'])
        data['y'] = torch.from_numpy(data['y'])
        data['e'] = torch.from_numpy(data['e'])         
    
        return data, None, None, None, data_in
    
    I_train, I_valid = validation_split(data['x'], val_fraction)    
    
    train_data = {}
    train_data['x'] = data['x'][I_train,:]
    train_data['t'] = data['t'][I_train]
    train_data['e'] = data['e'][I_train]
    train_data['y'] = data['y'][I_train]
    train_data['t'] = train_data['t'].reshape(len(train_data['t']),1)
    train_data['y'] = train_data['y'].reshape(len(train_data['y']),1)
    train_data['e'] = train_data['e'].reshape(len(train_data['e']),1)    

    valid_data = {}
    valid_data['x'] = data['x'][I_valid,:]
    valid_data['t'] = data['t'][I_valid]
    valid_data['e'] = data['e'][I_valid]
    valid_data['y'] = data['y'][I_valid]
    valid_data['t'] = valid_data['t'].reshape(len(valid_data['t']),1)
    valid_data['y'] = valid_data['y'].reshape(len(valid_data['y']),1)
    valid_data['e'] = valid_data['e'].reshape(len(valid_data['e']),1) 
    treatment_levels = np.unique(data['t'])
            
    return train_data, valid_data, treatment_levels, covariates, data_in 