import torch.nn as nn
import numpy as np
from .encoder import Encoder
from .decoder_t import *
from .decoder_e import *
from .decoder_ite import *
from utils import helpers
import torch
import pandas as pd
import torch.nn.functional as F


class DIR(nn.Module):
    def __init__(self, input_dim, treatment_levels, args, model_init):
        nn.Module.__init__(self)
        self.args = args
        self.treatment_levels = np.sort(treatment_levels)
        self.treatment_levels = self.treatment_levels.astype(int)
        self.t_loss_criterion = nn.CrossEntropyLoss()
        self.e_loss_criterion = nn.CrossEntropyLoss()
        self.encoder = Encoder(input_dim, args.hidden_dim, args.drop_in, args.hidden_layer_num, model_init)
        self.decoder_t = DecoderT(args.hidden_dim, len(treatment_levels), model_init)
        self.decoder_e = DecoderE(args.hidden_dim, model_init)
        self.decoder_y = DecoderY(args.hidden_dim, args.dim_out, args.drop_out, args.out_layer_num, self.treatment_levels, model_init)  
        self.separation = '_vv_'
        
        
    def reset_parameters(self, model_init):
        for param in self.parameters():
            model_init(param)
            
            
    def predict_ITE(self, l_ty, l_ey, l_y, t):
    
        return self.decoder_y (l_ty, l_ey, l_y, t)
        
    def forward(self, x, t):
        
        l_t, l_ty, l_e, l_ey, l_y = self.encoder(x) 
        pred_y = self.decoder_y (l_ty, l_ey, l_y, t)
        
        pred_t = self.decoder_t(l_t, l_ty)
        pred_e = self.decoder_e(l_e, l_ey, t)
        return pred_y, pred_t, pred_e, l_t, l_ty, l_e, l_ey, l_y 
        

    def do_prediction(self, x, factor):
            
        self.eval()  # enable eval model for turning off dropout, batch norm etc
        res = None
        liftScores = pd.DataFrame({})
        with torch.no_grad():
        

                
            for i in range(1, len(self.treatment_levels)): 
                t = torch.empty(x.shape[0], dtype=torch.int32 ).fill_(self.treatment_levels[i])
                t = t.view(-1,1)
                
                pred_ite, pred_log_t, pred_log_e, l_t, l_ty, l_e, l_ey, l_y = self.forward(x=x, t=t)
                
                
                
                probs_t = F.softmax(pred_log_t, dim=1)
        
                ## treatment_level must be zero based
                prob_val_t_current = probs_t[:, self.treatment_levels[i]]
                
                
                tempScore = pd.DataFrame((pred_ite.numpy() ) , columns =[factor + self.separation + str(self.treatment_levels[i])])
                liftScores = pd.concat([liftScores, tempScore], axis=1)
            
        return liftScores
            
        
    def do_train(self, iterator, optimizer, epoch):
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        # enable train model for turning on dropout, batch norm etc
        self.train()
        epoch_loss = 0.0
        epoch_t_ipm_loss = 0.0
        epoch_e_ipm_loss = 0.0
        epoch_y_risk_loss = 0.0
        epoch_t_prob_loss = 0.0
        epoch_e_prob_loss = 0.0
        
        running_loss = 0.0
        minNbr = 10
        for i, batch in enumerate(iterator):
        
            t = batch.t
            
            is_break = False
        
            for j in range(0, len(self.treatment_levels)):  
                tempT = t[t == self.treatment_levels[j]]
                
                if(tempT.shape[0] < minNbr):
                    is_break = True
                    break
                    
            if(is_break):
                continue
            
        
            # zero gradients buffers from previous batch in the
            optimizer.zero_grad()  
            
            # loss values: loss_y, loss_t, loss_e
            y_risk_loss, t_ipm_loss, e_ipm_loss, t_prob_loss, e_prob_loss = self._compute_loss(batch)
                       
            #loss = y_risk_loss + self.args.alpha1*t_ipm_loss + self.args.alpha2*t_ipm_loss+ self.args.beta1*t_prob_loss + self.args.beta2*e_prob_loss
            
            loss = y_risk_loss
            
            if(torch.isnan(loss)):
                print('Nan loss!!!!!!!!!!!!!!!! ==> skip')
                continue
            
            
            #################
            
            loss.backward()  # compute gradients
            optimizer.step()  # update model paramters 

            epoch_loss += loss.item() # sum loss value
            
            #################
            #epoch_t_ipm_loss += t_ipm_loss.item()  # sum loss value
            #epoch_e_ipm_loss += e_ipm_loss.item()
            epoch_y_risk_loss += y_risk_loss.item()
            epoch_t_prob_loss += t_prob_loss.item()
            epoch_e_prob_loss += e_prob_loss.item()
            
            
            
            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999: # print every 2000 mini-batches
              print('[%d, %5d] loss: %.3f' %
                   (epoch + 1, i + 1, running_loss/2000))
              running_loss = 0.0
      
        size = len(iterator)
        return epoch_loss / size, epoch_t_ipm_loss / size, epoch_e_ipm_loss / size, epoch_y_risk_loss / size, epoch_t_prob_loss / size, epoch_e_prob_loss / size

               
    
    def _get_sample_weight_treatment(self, pred_log_t):
    
        probs = F.softmax(pred_log_t, dim=1)
        if self.args.reweight_sample:       
            ''' Compute sample reweighting '''
            sample_weight = 1.*( 1. + (1.-self.pi_0)/self.pi_0 * (p_t/(1.-p_t))**(2.*t-1.) ) 
        else:
            sample_weight = 1.0
        return sample_weight
    
    def _compute_loss(self, batch):

        
        dindices = torch.argsort(batch.y[:,0], descending=True)
        x = batch.x
        y = batch.y
        e = batch.e
        t = batch.t
        c = batch.c 
        nc = batch.nc         
                
        # predict survival time y, treatment t, censoring e
        pred_ite, pred_log_t, pred_log_e, l_t, l_ty, l_e, l_ey, l_y = self.forward(x=x, t=t)
        
        
        for i in range(1, len(self.treatment_levels)): 
            t_tempt = torch.empty(x.shape[0], dtype=torch.int32 ).fill_(self.treatment_levels[i])
            t_tempt = t_tempt.view(-1,1)
            
            pred_ite_tempt = self.predict_ITE(l_ty, l_ey, l_y, t_tempt)
            
            pred_ite_tempt = pred_ite_tempt.view(-1,1)
            
            if(i > 1):
                pred_ite = torch.cat((pred_ite, pred_ite_tempt), dim=1)
            else:
                pred_ite = pred_ite_tempt
          
        
        # covariate discrepancy for treatment t, e
        #t_ipm_loss = helpers.compute_ipm(l_ty, t, self.treatment_levels)
        #e_ipm_loss = helpers.compute_ipm(l_ey, t, self.treatment_levels) 
        
        t_ipm_loss = helpers.mmd2_lin(l_ty, t, 0.5, self.treatment_levels)   
        e_ipm_loss = helpers.mmd2_lin(l_ey, e, 0.5, [0,1])        
        
        
        t_prob_loss = self.t_loss_criterion(pred_log_t, torch.tensor(t).long().squeeze())
        e_prob_loss = self.e_loss_criterion(pred_log_e, torch.tensor(e).long().squeeze()) 
       

        probs_t = F.softmax(pred_log_t, dim=1)
        
        ## treatment_level must be zero based
        prob_val_t = np.array([ probs_t[index_trea, t[index_trea]].detach().numpy()[0] for index_trea in range(len(t)) ])
        prob_val_t[prob_val_t < 0.001] = 0.001
        
        probs_e = F.softmax(pred_log_e, dim=1)
        
        prob_val_e = np.array([ probs_e[index_trea, e[index_trea]].detach().numpy()[0] for index_trea in range(len(e)) ])
        
        prob_val_e[prob_val_e < 0.001] = 0.001

        y_risk_loss = helpers.risk_lossSSP(t.squeeze(), pred_ite, torch.tensor(e).long().squeeze(), y.squeeze(), torch.tensor(prob_val_t), torch.tensor(1-prob_val_e), self.treatment_levels, self.args.zscore )
       
        
        return y_risk_loss, t_ipm_loss, e_ipm_loss, t_prob_loss, e_prob_loss
    