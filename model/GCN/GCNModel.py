import torch
import torch.nn.functional as F
from torch.nn import LayerNorm
from torch_geometric.nn import (
    GCNConv,
    GENConv,
    BatchNorm,
    global_mean_pool,
    global_max_pool,
    JumpingKnowledge,
    DeepGCNLayer,
)

class GCN(torch.nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=3, output_dim=1):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = GENConv(
                hidden_dim, hidden_dim, aggr='softmax', t=1.0, learn_t=True, num_layers=2
            )
            norm = LayerNorm(hidden_dim)
            act = torch.nn.PReLU(hidden_dim)
            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.2)
            self.layers.append(layer)
        self.jump = JumpingKnowledge(mode='cat')
        self.lin1 = torch.nn.Linear(hidden_dim * (num_layers + 1) * 2, hidden_dim * 2)
        self.lin2 = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.lin_out = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        xs = [x]
        for layer in self.layers:
            x = layer(x, edge_index)
            xs.append(x)
        x = self.jump(xs)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin_out(x)
        return x