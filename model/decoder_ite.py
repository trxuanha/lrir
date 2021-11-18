import torch.nn as nn
import numpy as np
import torch


class DecoderY(nn.Module):

    def __init__(self, hidden_dim, dim_out, dropout, layer_num, treatment_levels, model_init):
        nn.Module.__init__(self)
        
        self.outputName = 'output'
        
        self.treatment_levels = treatment_levels
        
        for i in range(1, len(self.treatment_levels)):

            print('treament level')
            print(self.treatment_levels[i])
            self.add_module( self.outputName + '_level_' + str(self.treatment_levels[i]), self._build_output_layer(3*hidden_dim, dim_out, dropout, layer_num))
        
        self.reset_parameters(model_init)

    def _getNonlinear(self, nlName):
        return nn.LeakyReLU()
        
    def _build_output_layer(self, dim_in, dim_out, dropout, layer_num):
        
        sizes = [dim_in] + [dim_out] * (layer_num - 1)
        isFirst = True
        fc = nn.Sequential()
        
        for in_size, out_size in zip(sizes, sizes[1:]):
            if(isFirst):
                fc.add_module(str(len(fc)), nn.Linear(in_features=dim_in, out_features=dim_out))
                isFirst = False
            else:
                fc.add_module(str(len(fc)), nn.Linear(in_features=dim_out, out_features=dim_out)) 
            fc.add_module(str(len(fc)), self._getNonlinear(None))
            fc.add_module(str(len(fc)), nn.Dropout(dropout))   

        fc.add_module(str(len(fc)), nn.Linear(in_features=dim_out, out_features=1))  
        
        return fc
            
    def reset_parameters(self, model_init):
        for param in self.parameters():
            model_init(param)

    def forward(self, l_ty, l_ey, l_y, t):
        phi = torch.cat((l_ty, l_ey, l_y), dim=1)
        res  = torch.rand(t.shape[0])
        
            
        for i in range(1, len(self.treatment_levels)):
            sel_index = t==self.treatment_levels[i]
            
            sel_phi = phi[sel_index.squeeze(),:]
            if(sel_phi.shape[0] == 0):
                continue
                         
            outFunc = self.get_submodule(self.outputName + '_level_' + str(self.treatment_levels[i]))
            h_hat = outFunc(sel_phi) 
            res[sel_index.squeeze()] = h_hat.squeeze()

        return res
