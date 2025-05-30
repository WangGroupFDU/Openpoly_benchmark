import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, BatchNorm

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, BatchNorm

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, num_heads=4):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, concat=True)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, concat=True)
        self.conv3 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, concat=False)
        self.fc = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        x = self.conv1(x, edge_index)
        x = F.elu(x)

        x = self.conv2(x, edge_index)
        x = F.elu(x)

        x = self.conv3(x, edge_index)
        x = F.elu(x)

        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=0.5, training=self.training)

        x = self.fc(x)

        return x