import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from torch.nn.parameter import Parameter
import torch
import numpy as np


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nL):
        super(GCN, self).__init__()
        self.nL = nL
        self.gc1 = GraphConvolution(nfeat, nhid)
        for i in range(self.nL):
            name = "gcn_for{}".format(i+1)
            self.add_module(name, GraphConvolution(nhid, nhid))
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        temp = [x]
        for i, Module in enumerate(self.modules()):
            if i == 0:
                continue
            elif i == 1:
                x = F.relu(Module(temp[-1], adj))
                x = F.dropout(x, self.dropout, training=self.training)
                temp.append(x)
            elif i == (self.nL + 2): 
                x = Module(temp[-1], adj)
                break
            elif i < 3 :
                x = F.relu(Module(temp[-1], adj))
                temp.append(x)
            elif i < (self.nL + 2):
                x = F.relu(Module(temp[-1], adj)) + temp[-2]
                temp.append(x)
        return F.log_softmax(x, dim=1)
