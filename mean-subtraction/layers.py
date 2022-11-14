import math

import torch
import torch.nn as nn

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.count = 0
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        # self.ln = nn.LayerNorm(in_features)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, feature, adj, mode):
        if mode in [0, 3, 4]:
            support = torch.spmm(adj, feature)
            output = torch.mm(support,  self.weight)
        elif mode == 1:
            feature = feature - torch.mean(feature, dim=0, keepdim=True)
            support = torch.spmm(adj, feature)
            output = torch.mm(support,  self.weight)
        elif mode == 2:
            feature = feature - torch.mean(feature, dim=0, keepdim=True)
            feature = torch.mm(feature, torch.diag(torch.flatten(torch.std(feature, dim=0) + 0.0001) ** (-1)))
            support = torch.spmm(adj, feature)
            output = torch.mm(support,  self.weight)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
