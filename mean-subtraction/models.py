import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from torch.nn.parameter import Parameter
import torch
import numpy as np


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nL, mode):
        super(GCN, self).__init__()
        self.nL = nL
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.mode = mode
        for i in range(self.nL):
            name = "gcn_for{}".format(i+1)
            self.add_module(name, GraphConvolution(nhid, nhid))
            if self.mode == 3:
                name = "bn_for{}".format(i+1)
                self.add_module(name, nn.BatchNorm1d(nhid))
            elif self.mode == 4:
                name = "ln_for{}".format(i+1)
                self.add_module(name, nn.LayerNorm(nhid))
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        if self.mode in [0, 1, 2]:
            temp = [x]
            for i, Module in enumerate(self.modules()):
                if i == 0:
                    continue
                elif i == 1:
                    x = F.relu(Module(temp[-1], adj, self.mode))
                    x = F.dropout(x, self.dropout, training=self.training)
                    temp.append(x)
                elif i == (self.nL + 2): 
                    x = Module(temp[-1], adj, self.mode)
                    break
                elif i < 3 : # i < 4
                    x = F.relu(Module(temp[-1], adj, self.mode))
                    temp.append(x)
                elif i < (self.nL + 2):
                    x = F.relu(Module(temp[-1], adj, self.mode)) + temp[-2] # + temp[-3]
                    temp.append(x)
        elif self.mode in [3, 4]:
            temp = [x]
            for i, Module in enumerate(self.modules()):
                if i == 0:
                    continue
                elif i == 1:
                    x = F.relu(Module(temp[-1], adj, self.mode))
                    x = F.dropout(x, self.dropout, training=self.training)
                    temp.append(x)
                elif i % 2 == 1:
                    try:
                        x = F.relu(Module(x)) + temp[-2] # + temp[-3]
                    except:
                        x = F.relu(Module(x))
                    temp.append(x)
                elif i < 3 * 2:
                    x = Module(temp[-1], adj, self.mode)
                elif i == (2 * self.nL + 1): 
                    x = Module(temp[-1], adj, self.mode)
                    break
                elif i < (2 * self.nL + 1):
                    x = Module(temp[-1], adj, self.mode)
        return F.log_softmax(x, dim=1)
