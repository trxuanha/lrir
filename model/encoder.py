import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout, layer_num, model_init):
        nn.Module.__init__(self)
        self.hidden_dim = hidden_dim
        # construct hidden layers
        self.l_t = self._build_latent_layer(input_dim, hidden_dim, dropout, layer_num)
        self.l_ty = self._build_latent_layer(input_dim, hidden_dim, dropout, layer_num)
        self.l_e = self._build_latent_layer(input_dim, hidden_dim, dropout, layer_num)
        self.l_ey = self._build_latent_layer(input_dim, hidden_dim, dropout, layer_num)
        self.l_y = self._build_latent_layer(input_dim, hidden_dim, dropout, layer_num)
        self.reset_parameters(model_init)
        
    def _getNonlinear(self, nlName):
        return nn.LeakyReLU()
        
        
    def _build_latent_layer(self, dim_in, dim_out, dropout, layer_num):
    
        epsilon = 1e-5
        alpha = 0.9  # use numbers closer to 1 if you have more data
        
        sizes = [dim_in] + [dim_out] * (layer_num - 1)
        fc = nn.Sequential()
        isFirst = True
        
        for in_size, out_size in zip(sizes, sizes[1:]):
                    
            if(isFirst):
                fc.add_module(str(len(fc)), nn.Linear(in_features=dim_in, out_features=dim_out))
                isFirst = False
            else:
                fc.add_module(str(len(fc)), nn.Linear(in_features=dim_out, out_features=dim_out))    
            fc.add_module(str(len(fc)), nn.BatchNorm1d(dim_out, eps=epsilon, momentum=alpha))
            fc.add_module(str(len(fc)), self._getNonlinear(None))
            fc.add_module(str(len(fc)), nn.Dropout(dropout))
        
        return fc

    def reset_parameters(self, model_init):
        for param in self.parameters():
            model_init(param)

    def forward(self, x):
                     
        l_t = self.l_t (x) 
        l_ty = self.l_ty (x)
        l_e = self.l_e (x) 
        l_ey = self.l_ey (x)
        l_y = self.l_y (x)         
            
        return l_t, l_ty, l_e, l_ey, l_y