import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPRegression(nn.Module):
    def __init__(self, hidden_dim1: int = 256, output_dim: int = 1):
        super(MLPRegression, self).__init__()
        self.fc1 = nn.LazyLinear(hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, output_dim)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))  
        x = self.dropout(x)     
        out = self.fc2(x)       
        return out