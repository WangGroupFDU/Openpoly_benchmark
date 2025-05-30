import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.3):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_size 
        

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_size))     
            layers.append(nn.BatchNorm1d(hidden_size))        
            layers.append(nn.ReLU())                              
            layers.append(nn.Dropout(dropout_rate))            
            in_dim = hidden_size                                 

        layers.append(nn.Linear(in_dim, output_size))
        
        self.network = nn.Sequential(*layers)

    def forward(self, data):

        x = data.morgan_fp  
        out = self.network(x)  
        return out