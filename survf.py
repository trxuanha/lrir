from model.dir import DIR
from utils import helpers, custom_batch
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
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from pysurvival.models.simulations import SimulationModel
from pysurvival.models.survival_forest import RandomSurvivalForestModel
from pysurvival.utils.metrics import concordance_index
from pysurvival.utils.display import integrated_brier_score



#https://square.github.io/pysurvival/models/nonlinear_coxph.html

def validation_split(D_exp, val_fraction):
    """ Construct a train/validation split """
    n = D_exp.shape[0]

    if val_fraction > 0:
        n_valid = int(val_fraction*n)
        n_train = n-n_valid
        I = np.random.permutation(range(0,n))
        I_train = I[:n_train]
        I_valid = I[n_train:]
    else:
        I_train = range(n)
        I_valid = []

    return I_train, I_valid


def load_data(fname, factor, target, allFactors, exVars, event, val_fraction=None, icovariates=None):

    """ Load data set """
    data_in = pd.read_csv(fname,  encoding = "ISO-8859-1", engine='python')
    covariates = list(set(data_in.columns) - {target, event} - set(exVars) - set(allFactors))
    
    if(icovariates!= None):
        covariates = icovariates
        
    print('covariates')
    
    
    print(covariates)
    
    data = {}
    data['x'] = data_in[covariates].to_numpy().astype(np.float64)
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
    
        return data, None, None, None
    
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
            
    return train_data, valid_data, treatment_levels, covariates 
    
    
    
def init_config():
    
    parser = argparse.ArgumentParser(description='Causal Survival Analysis')
    parser.add_argument('--hidden_dim', type=int, default= 100)
    parser.add_argument('--drop_in', type=float, default= 0.2)
    parser.add_argument('--dim_out', type=int, default= 100)
    parser.add_argument('--drop_out', type=float, default= 0.2)
    parser.add_argument('--batch_size', type=int, default= 100)
    parser.add_argument('--iterations', type=int, default= 1000)
    parser.add_argument('--datadir', type=str, default= 'input')
    parser.add_argument('--outdir', type=str, default= 'output')
    parser.add_argument('--input_name', type=str, default= 'employment')
    parser.add_argument('--learning_rate', type=float, default= 3e-4)  #
    parser.add_argument('--target', type=str, default= 'EMP_PERIOD')
    parser.add_argument('--all_factors', type=str, default= 'MOTIVATION,NBRSUPPORT,WORK_TYPE')
    parser.add_argument('--excl_factors', type=str, default= 'WORK_ENV,PER_REA,EXPECT,LOW_PER')
    parser.add_argument('--event', type=str, default= 'EVENT')
    parser.add_argument('--val_part', type=float, default= 0.05)
    parser.add_argument('--hidden_layer_num', type=int, default= 5)
    parser.add_argument('--out_layer_num', type=int, default= 3)
    parser.add_argument('--alpha1', type=float, default= 1)
    parser.add_argument('--alpha2', type=float, default= 1)
    parser.add_argument('--beta1', type=float, default= 1)
    parser.add_argument('--beta2', type=float, default= 1)
    parser.add_argument('--plambda', type=float, default= 1e-10)
    parser.add_argument('--seed', type=int, default= 324)
    parser.add_argument('--reweight_sample', type=int, default= 0)
    parser.add_argument('--config_num', type=int, default= 1)
    parser.add_argument('--fold_num', type=int, default= -1)
    parser.add_argument('--zscore', type=float, default= 3)
    args = parser.parse_args()
    return args


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
    
    
    
def estimateUplift(estimatedImprovements, outcomeName, sortbyabs=False):
    
    if(sortbyabs):
        estimatedImprovements['ABS_Improvement'] = estimatedImprovements['LIFT_SCORE'].abs()
        estimatedImprovements.sort_values(by=['ABS_Improvement'], ascending = [False], inplace=True, axis=0)
    else:
        estimatedImprovements.sort_values(by=['LIFT_SCORE'], ascending = [False], inplace=True, axis=0)
    estimatedImprovements = estimatedImprovements.reset_index(drop=True) 
    
    Sum_Y_Follow_Rec    = np.array([])
    Sum_Nbr_Follow_Rec    = np.array([])
    Sum_Y_Not_Follow_Rec    = np.array([])
    Sum_Nbr_Not_Follow_Rec    = np.array([])
    Improvement    = np.array([])
    total_Y_Follow_Rec  = 0
    total_Nbr_Follow_Rec = 0
    total_Y_Not_Follow_Rec  = 0
    total_Nbr_Not_Follow_Rec = 0    
    for index, individual in estimatedImprovements.iterrows():
        improvementTemp = 0
        if(individual['FOLLOW_REC'] == 1):
            total_Nbr_Follow_Rec = total_Nbr_Follow_Rec + 1
            total_Y_Follow_Rec = total_Y_Follow_Rec + individual[outcomeName]
        else:
            total_Nbr_Not_Follow_Rec = total_Nbr_Not_Follow_Rec + 1
            total_Y_Not_Follow_Rec = total_Y_Not_Follow_Rec + individual[outcomeName]   
        Sum_Nbr_Follow_Rec = np.append (Sum_Nbr_Follow_Rec, total_Nbr_Follow_Rec)
        Sum_Y_Follow_Rec = np.append (Sum_Y_Follow_Rec, total_Y_Follow_Rec)
        Sum_Nbr_Not_Follow_Rec = np.append (Sum_Nbr_Not_Follow_Rec, total_Nbr_Not_Follow_Rec)
        Sum_Y_Not_Follow_Rec = np.append (Sum_Y_Not_Follow_Rec, total_Y_Not_Follow_Rec)
        if(total_Nbr_Follow_Rec == 0 or total_Nbr_Not_Follow_Rec == 0 ):
            if(total_Nbr_Follow_Rec > 0):
                improvementTemp = (total_Y_Follow_Rec/total_Nbr_Follow_Rec)
            else:
                improvementTemp = 0
        else:
            improvementTemp = (total_Y_Follow_Rec/total_Nbr_Follow_Rec) - (total_Y_Not_Follow_Rec/total_Nbr_Not_Follow_Rec)   
        Improvement = np.append (Improvement, improvementTemp)
    ser = pd.Series(Sum_Nbr_Follow_Rec)
    estimatedImprovements['N_TREATED'] = ser
    ser = pd.Series(Sum_Y_Follow_Rec)
    estimatedImprovements['Y_TREATED'] = ser
    ser = pd.Series(Sum_Nbr_Not_Follow_Rec)
    estimatedImprovements['N_UNTREATED'] = ser
    ser = pd.Series(Sum_Y_Not_Follow_Rec)
    estimatedImprovements['Y_UNTREATED'] = ser  
    ser = pd.Series(Improvement)
    estimatedImprovements['UPLIFT'] = ser
    return estimatedImprovements


def findRec(currentRow):
    separation = '_vv_'
    bestFactor = currentRow['TREATMENT_NAME']    
    res = bestFactor.split(separation)    
    facName = res[0]
    facVal = res[1]    
    return 1 if str(currentRow[facName]) == str(facVal) else 0
      
      
     

    
def wrapperPara(x, includeCen=False):

    fold_count = x
    args = init_config()
    
    method = 'Survf'
    fold = 5
    BASE_DIR = os.getcwd()
    inputPath = os.path.join(BASE_DIR, 'input', args.input_name, 'split')
    testPath = os.path.join(BASE_DIR, 'input', args.input_name, 'split')
    
    
    ### Set random seed for determinstic result
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(seed=args.seed)


    
    
    target = args.target
    allFactors = args.all_factors.split(',')
    exVars = args.excl_factors.split(',')
    event = args.event
    separation = '_vv_'
    
    globalLiftScores = pd.DataFrame({})
    for factor in allFactors:
    
        outdir = os.path.join(BASE_DIR, 'output', method, args.input_name, factor)
        if not os.path.exists(outdir):    
            os.mkdir(outdir)    
        
        # Load data
        datapath = os.path.join(inputPath, args.input_name + '_train_' + str(fold) + '.csv')
        train_data, valid_data, treatment_levels, covariates = load_data(datapath, factor, target, allFactors, exVars, event, args.val_part)
        
        testFilepath = os.path.join(testPath, args.input_name + '_test_'  +  str(fold) + '.csv')
        test_data, tempD1, tempD2, tempD3  = load_data(testFilepath, factor, target, allFactors, exVars, event, None, covariates)
        
       
        rsf = RandomSurvivalForestModel(num_trees=500)
  
        t = np.reshape(train_data['t'],(1, train_data['t'].size))
        X = np.concatenate((train_data['x'], t.T), axis=1)
        
        rsf.fit(X, train_data['y'].flatten(), train_data['e']. flatten(),
                max_features="sqrt", max_depth=5, min_node_size=20)        
        
        liftScores = pd.DataFrame({})
        
        t = np.empty(test_data['x'].shape[0], dtype=int )
        t.fill(treatment_levels[0])
        t = np.reshape(t,(1, t.size))
        
        
        x_new = np.concatenate((test_data['x'], t.T), axis=1)
        
        risks0 = rsf.predict_risk(x_new)
        
        
        for i in range(1, len(treatment_levels)): 
            t = np.empty(test_data['x'].shape[0], dtype=int )
            t.fill(treatment_levels[i])
            t = np.reshape(t,(1, t.size))
            x_new = np.concatenate((test_data['x'], t.T), axis=1)            
                
            risks1 = rsf.predict_risk(x_new)
            
            tempScore = pd.DataFrame(-(risks1-risks0) , columns =[factor + separation + str(treatment_levels[i])])
            liftScores = pd.concat([liftScores, tempScore], axis=1)

        globalLiftScores = pd.concat([globalLiftScores, liftScores], axis=1) 
        
        
        test_data_f = pd.read_csv(testFilepath,  encoding = "ISO-8859-1", engine='python')
        
        test_data_f['LIFT_SCORE'] = liftScores.max(axis=1)
        test_data_f['TREATMENT_NAME'] = liftScores.idxmax(axis=1)
        test_data_f = pd.concat((test_data_f, liftScores), axis=1)

        test_data_f['FOLLOW_REC'] = test_data_f.apply(lambda x: 1 if x['TREATMENT_NAME'] ==  factor + separation + str(int(x[factor]))  else 0 , axis=1)
        
        if(not includeCen):
            print('Include complete data')
            test_data_f = test_data_f[test_data_f[event] == 1]
        else:
            print('Include censoring data')
        
        test_data_f = estimateUplift(test_data_f, target)
        
        if(includeCen):
            outFilePath = os.path.join(outdir, args.input_name +'_' + method + '_Cen_' + str(fold_count) + '.csv')
        else:
            outFilePath = os.path.join(outdir, args.input_name +'_' + method + '_' + str(fold_count) + '.csv')
            
        test_data_f.to_csv(outFilePath, index=False)
        
    ## end for factor loop
    
    outdirb = os.path.join(BASE_DIR, 'output', method, args.input_name, 'bestFactor')
    if not os.path.exists(outdirb):    
        os.mkdir(outdirb) 
        
    testFilepath = os.path.join(testPath, args.input_name + '_test_'  +  str(fold) + '.csv')
    best_data_f = pd.read_csv(testFilepath,  encoding = "ISO-8859-1", engine='python')
    best_data_f ['LIFT_SCORE'] = globalLiftScores.max(axis=1)   
    best_data_f['TREATMENT_NAME'] = globalLiftScores.idxmax(axis=1)
    best_data_f = pd.concat([best_data_f, globalLiftScores], axis=1)
    best_data_f['FOLLOW_REC'] = best_data_f.apply(findRec, axis=1)
    ## only select observed data
    if(not includeCen):
        print('Include complete data')
        best_data_f = best_data_f[best_data_f[event] == 1]
    else:
        print('Include censoring data')  
    
    best_data_f = estimateUplift(best_data_f, target)
    if(includeCen):
        outFilePath = os.path.join(outdirb, args.input_name +'_' + method + '_Cen_' + str(fold_count) + '.csv')
    else:
        outFilePath = os.path.join(outdirb, args.input_name +'_' + method + '_' + str(fold_count) + '.csv')
        
    best_data_f.to_csv(outFilePath, index=False) 
   
        
foldNbr = [1,2,3,4,5]  
    
if __name__ == '__main__':
    
    args = init_config()
    fold_num = args.fold_num
    
    if(fold_num > 0):
        wrapperPara(fold_num)
    else:    
        for i in range(0, len(foldNbr)):
            wrapperPara(foldNbr[i])
            wrapperPara(foldNbr[i], True)
    #with Pool(5) as p:
    #   p.map(wrapperPara, foldNbr)
    #   p.close()