import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.dense import DenseGCNConv

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, to_dense_adj, degree
from torch.nn import Linear as Lin
from torch_geometric.nn import SAGEConv
import numpy as np

# pass messages between different genes in same cell
class RegGNN(MessagePassing):
    def __init__(self, in_channels,hidden , out_channels, aggr='max', node_dim = -2, bias = True,
                     **kwargs):
        super(RegGNN, self).__init__(aggr=aggr, **kwargs
        self.lin1 = Lin(2*110, 110, bias=bias)
        self.lin2 = Lin(in_channels, hidden, bias=bias)
        self.lin3 = Lin(110, 110, bias=bias)

    def forward(self, x, edge_index, edge_weight=None, size=None):
        h = F.relu(self.lin2(x.t()))
        h = h.t()
        edge_index = edge_index.int()
        edge_index = edge_index.nonzero().contiguous()
        edge_index = edge_index.t()
        
        x = self.propagate(edge_index, size=size, x=x, h=h,
                              edge_weight=edge_weight)
        return x

    def message(self, h_j, edge_weight):
        return h_j if edge_weight is None else edge_weight.view(-1, 1) * h_j

    def update(self, aggr_out, h):
        return self.lin1(torch.cat((h.t(), aggr_out.t()), 1)).t()

    def loss(self, pred, score):
        return F.mse_loss(pred, score)

# message passing and then fully connected layer to predict protein expression
class RegGNN3(nn.Module):
    def __init__(self, in_channels):
        super(RegGNN3, self).__init__()

        # self.conv1 = FeatGraphConv(in_channels, in_channels, in_channels, aggr='mean')
        self.conv1 = DenseGCNConv(in_channels,in_channels)
        self.lin = Lin(110, 110)

    def forward(self, x, edge_index):
        x = x.t()
        x = self.lin(x)
        x = x.t()
        return x

    def loss(self, pred, score):
        return F.mse_loss(pred, score)
