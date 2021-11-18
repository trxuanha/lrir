import torch.nn as nn
import torch
import numpy as np
from scipy import stats
import random

from utils.ot_compute import SinkhornDistance
from lifelines import KaplanMeierFitter


def estimate_km(y, e):
    kmf = KaplanMeierFitter()
    kmf.fit(durations=y, event_observed=e)
    return kmf
    
    
### calculate the number of trainable parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class uniform_initializer(object):
    def __init__(self, stdv):
        self.stdv = stdv
        random.seed(20)
        np.random.seed(20)
        torch.manual_seed(20)

    def __call__(self, tensor):
        nn.init.uniform_(tensor, -self.stdv, self.stdv)


use_cuda = torch.cuda.is_available()


def gData(data):
    tensor = data
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    if use_cuda:
        tensor = tensor.cuda()
    return tensor


def gVar(data):
    return gData(data)
    
def compute_ipm(X, t, treatment_levels, criteria=None):
    ipm = 0
    Xc = None
    Xt = None 
    # small epsilon is close to the true wasserstein distance
    
    for i in range(0, len(treatment_levels)):   
        if(i == 0):
            ic = t == treatment_levels[i]
            Xc = X[ic.squeeze(),:]
        else:
            it = t == treatment_levels[i]
            Xt = X[it.squeeze(),:]
            sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, reduction=None)
            dist, P, C = sinkhorn(x=Xc, y=Xt)
            
            
            ipm = ipm + dist

    return torch.tensor(ipm, dtype=torch.double)
    

def mmd2_lin(X,t,p, treatment_levels):

    ipm = 0
    Xc = None
    Xt = None 
    mean_control = 0
    mean_treated = 0
    for i in range(0, len(treatment_levels)):   
        if(i == 0):
            ic = t == treatment_levels[i]
            Xc = X[ic.squeeze(),:]
            mean_control = torch.mean(Xc, dim=0)
        else:
            it = t == treatment_levels[i]
            Xt = X[it.squeeze(),:]
            mean_treated = torch.mean(Xt,dim=0)
            
            dist = torch.sum(torch.square(2.0*p*mean_treated - 2.0*(1.0-p)*mean_control))
                    
            ipm = ipm + dist

    return ipm
    
    
    
    
def risk_loss(pred_y, e, y, weight_t):
    
    
    loss = weight_t*torch.square(y - pred_y)*e
        
    return torch.sum(loss) 


    
def estimate_ATE(t, e, y, prob_t, prob_c, treatment_levels):

    # baseline level
    
    sel_index = t == treatment_levels[0]
    sel_index = sel_index.squeeze()
    
    base_outcome_adj = e[sel_index] * y[sel_index]/(prob_t[sel_index]* prob_c[sel_index])
    base_weight = e[sel_index]/(prob_t[sel_index]* prob_c[sel_index])
    

    res = np.array([])
    
    for i in range(1, len(treatment_levels)):
    
        sel_index = t == treatment_levels[i]
        sel_index = sel_index.squeeze()
        
        sel_e = e[sel_index]    
        sel_e = sel_e[sel_e == 1]
        
        treat_outcome_adj = e[sel_index] * y[sel_index]/(prob_t[sel_index]* prob_c[sel_index])
        treat_weight = e[sel_index]/(prob_t[sel_index]* prob_c[sel_index])        
        ipwEffect = (torch.sum(treat_outcome_adj)/torch.sum(treat_weight)) - (torch.sum(base_outcome_adj)/torch.sum(base_weight))
        res = np.append (res, ipwEffect)
        
        
    return res
    

def risk_lossSSP(t, pred_ite, e, y, prob_t, prob_c, treatment_levels, zscore ):    



    ate_n = estimate_ATE(t, e, y, prob_t, prob_c, treatment_levels)
    
    m_t = np.ma.array(t, mask=False)
    
    smNbr = t.shape[0]
    groundTruth  = np.empty([t.shape[0], len(treatment_levels) -1])
    groundTruth[:] = np.NaN
    for i in range(smNbr):
    
        # exclude elemment i
        m_t.mask[i] = True    
            
        if(e[i] == 0):
            m_t.mask[i] = False
            continue

        
        sel_index = ~m_t.mask
                
        ate_n_ex = estimate_ATE(t[sel_index], e[sel_index], y[sel_index], prob_t[sel_index], prob_c[sel_index], treatment_levels)
        
        ITE = smNbr*(ate_n - ate_n_ex) + ate_n_ex
              
        groundTruth[i,:] = ITE
        
        m_t.mask[i] = False
        

    z_scoreThreshold = zscore # turnover, employment, ACTG175(300)
    #z_scoreThreshold = 4 # 
    offset = 0.9
    
    # offset = 0.8, gbsg, turnover, hr, 
    
    dup_e= np.tile(e.view(-1,1), (1, len(treatment_levels) -1))
    #groundTruth[ groundTruth > ate_n*(1 + offset) ] = np.nan
    #groundTruth[ groundTruth < ate_n* (1 - offset) ] = np.nan
    
    z_scores = np.abs(stats.zscore(groundTruth, nan_policy='omit'))
    # It is crucial to select good training samples
    groundTruth[ z_scores > z_scoreThreshold] = np.nan
    
    
    dup_e[np.isnan(groundTruth)] = 0
    groundTruth[np.isnan(groundTruth)] = 0
    
    loss_max = (torch.tensor(groundTruth) - pred_ite)*torch.tensor(dup_e) 
    loss = torch.square(loss_max)
    
    
    return torch.sum(loss) 
    

        
