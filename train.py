from model.dir_ssp import DIR
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
#import torch.multiprocessing as Pool

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
    data['x'] = data_in[covariates].to_numpy()
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
    
        return data, None, None, None, None
    
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
    
    #train_data['c'] = np.random.rand(train_data['e'].shape[0], 1)
    
    train_data['c'] = np.full(train_data['e'].shape, 10.1)
    train_data['nc'] = np.full(train_data['e'].shape, 10.1)
    
    kmf_moldes = []
    for i in range(0, len(treatment_levels)):
    
        sel_index = train_data['t'] == treatment_levels[i]
    
        c_kmf = helpers.estimate_km(y=train_data['y'][sel_index ], e=1 - train_data['e'][sel_index]) 
        c_prob = np.array(c_kmf.predict(train_data['y'][sel_index ]))
        c_prob[c_prob < 0.001] = 0.001
        train_data['c'][sel_index] = c_prob
        
        kmf_moldes.append(c_kmf)
        
        c_kmf = helpers.estimate_km(y=train_data['y'][sel_index ], e= train_data['e'][sel_index]) 
        c_prob = np.array(c_kmf.predict(train_data['y'][sel_index ]))
        c_prob[c_prob < 0.001] = 0.001
        train_data['nc'][sel_index] = c_prob        
        
        

    return train_data, valid_data, treatment_levels, covariates, kmf_moldes 
    
    
    
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
      
    
def wrapperPara(x):

    fold_count = x
    args = init_config()
    
    method = 'LRIR'
    fold = fold_count
    BASE_DIR = os.getcwd()
    inputPath = os.path.join(BASE_DIR, 'input', args.input_name, 'split')
    testPath = os.path.join(BASE_DIR, 'input', args.input_name, 'split')
    
    log_file = 'logs/args.{}_fold_{}.log'.format(args.input_name, x)
    logging.basicConfig(filename=log_file, filemode='w', level=logging.DEBUG)
    logging.debug(args)  
    
    model_init = helpers.uniform_initializer(0.01)
    
    ### Set random seed for determinstic result
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(seed=args.seed)

    ### Torch device for putting tensors into GPU if available
    
    
    cuda_device = torch.device('cuda')
    cpu_device = torch.device('cpu')
    cuda_tensor = 'torch.cuda.DoubleTensor'
    cpu_tensor = 'torch.DoubleTensor'
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = cuda_device
        torch.set_default_tensor_type(cuda_tensor)
        torch.cuda.manual_seed(seed=args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.set_default_tensor_type(cpu_tensor)
        device = cpu_device
        
    
    
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
        train_data, valid_data, treatment_levels, covariates, kmf_moldes = load_data(datapath, factor, target, allFactors, exVars, event, args.val_part)
        
        testFilepath = os.path.join(testPath, args.input_name + '_test_'  +  str(fold) + '.csv')
        test_data, tempD1, tempD2, tempD3, tempD4  = load_data(testFilepath, factor, target, allFactors, exVars, event, None, covariates)
        
        model_init = helpers.uniform_initializer(0.1)
        model = DIR(input_dim=train_data['x'].shape[1], treatment_levels=treatment_levels, args=args, model_init=model_init)
        
        print('****Processing for factor****: {}'.format(factor))
        print(args)
        #print(model)
        parameters = helpers.count_parameters(model)
        print_param = "The model has trainable parameters:{}".format(parameters)
        #print(print_param)
        #logging.debug(print_param)        
        
        ### Optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.plambda)
        
        iterators = custom_batch.build_iterator(args=args, train_data=train_data, valid_data=valid_data, test_data=test_data)    
        
        icount = 0

        logging.debug('epoch loop')
        for epoch in range(args.iterations):
            start_time = time.time()
            
            #logging.debug('start epoch train: {}'.format(epoch + 1))
            train_loss, train_t_ipm_loss, train_e_ipm_loss, train_y_risk_loss, train_t_prob_loss, train_e_prob_loss  = model.do_train(iterator=iterators["train_iterator"],
                                                                            optimizer=optimizer, epoch=epoch)
                                                                            
            end_time = time.time()
            
            

            epoch_mins, epoch_sec = epoch_time(start_time=start_time, end_time=end_time)
            
            time_str = 'end epoch train: {}, Time: {}m {}s'.format(epoch + 1, epoch_mins, epoch_sec)
            #print(time_str)
            #logging.debug(time_str)
            
            icount = icount + 1
            if(icount % 100 == 9):
            #if(9 == 9):
                print_epoch = "Epoch:{} | Time: {}m {}s".format(epoch + 1, epoch_mins, epoch_sec)           
           
                print(print_epoch)
                logging.debug(print_epoch)
                
                print_train = "\t train_loss Loss:{} |  train_t_ipm_loss:{} |  train_e_ipm_loss:{} |train_y_risk_loss:{} |train_t_prob_loss:{} |train_e_prob_loss:{}".format(train_loss, train_t_ipm_loss, train_e_ipm_loss, train_y_risk_loss, train_t_prob_loss, train_e_prob_loss)
                
                print(print_train)
                logging.debug(print_train)
            
            
        liftScores  = model.do_prediction(test_data['x'], factor)
        globalLiftScores = pd.concat([globalLiftScores, liftScores], axis=1) 
            
        test_data_f = pd.read_csv(testFilepath,  encoding = "ISO-8859-1", engine='python')
        test_data_f['LIFT_SCORE'] = liftScores.max(axis=1)
        test_data_f['TREATMENT_NAME'] = liftScores.idxmax(axis=1)
        test_data_f = pd.concat([test_data_f, liftScores], axis=1)
        test_data_f['FOLLOW_REC'] = test_data_f.apply(lambda x: 1 if x['TREATMENT_NAME'] ==  factor + separation + str(int(x[factor]))  else 0 , axis=1)
        test_data_f = test_data_f[test_data_f[event] == 1]
        test_data_f = estimateUplift(test_data_f, target)
        
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
    best_data_f = best_data_f[best_data_f[event] == 1]
    best_data_f = estimateUplift(best_data_f, target)
    outFilePath = os.path.join(outdirb, args.input_name +'_' + method + '_' + str(fold_count) + '.csv')  
    best_data_f.to_csv(outFilePath, index=False)     
        
foldNbr = [1,2,3,4,5]  
    
if __name__ == '__main__':
    
    args = init_config()
    fold_num = args.fold_num
    
    torch.manual_seed(20)
    random.seed(20)
    np.random.seed(20)
    
    
    if(fold_num > 0):
        wrapperPara(fold_num)
    else:    
        for i in range(0, len(foldNbr)):
            wrapperPara(foldNbr[i])
    
    
    '''
    if(fold_num > 0):
        foldNbr = [fold_num] 
        
    with closing(Pool(5)) as pool:
    
        for i in range(0, len(foldNbr)):
            pool.apply_async(wrapperPara, [foldNbr[i]])
            
        pool.close()
        pool.terminate()
    '''