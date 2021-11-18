import torch.nn as nn
import torch.nn.functional as F
import torch


class DecoderT(nn.Module):

    def __init__(self, hidden_dim, treat_dim, model_init):
        nn.Module.__init__(self)
        self.linReg = nn.Linear(in_features=2*hidden_dim, out_features=treat_dim)
        self.reset_parameters(model_init)
            
    def reset_parameters(self, model_init):
        for param in self.parameters():
            model_init(param)

    def forward(self, l_t, l_ty):
        phi = torch.cat((l_t, l_ty), dim=1)        
        res = self.linReg(phi)
       
        return res
