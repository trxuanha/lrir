import torch.nn as nn
import torch


class DecoderE(nn.Module):

    def __init__(self, hidden_dim, model_init):
        nn.Module.__init__(self)
        self.linReg = nn.Linear(in_features=2*hidden_dim+1, out_features=2)
        self.reset_parameters(model_init)
            
    def reset_parameters(self, model_init):
        for param in self.parameters():
            model_init(param)

    def forward(self, l_e, l_ey, t):
        phi = torch.cat((t, l_e, l_ey), dim=1)        
        res = self.linReg(phi)
       
        return res
