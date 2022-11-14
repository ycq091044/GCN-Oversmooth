import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from torch.nn.parameter import Parameter
import torch
import numpy as np

def cosine(m):
    d = m.T @ m
    norm = (m * m).sum(0, keepdims=True) ** .5
    return d / norm / norm.T

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
        
        # FinalRep = temp[-1].cpu().detach().numpy()
        # nNum, fNum = FinalRep.shape
        # Fwise = np.abs(cosine(FinalRep + 1e-9 * np.random.randn(nNum, fNum))).sum() / fNum / fNum
        # Nwise = np.abs(cosine(FinalRep.T + 1e-9 * np.random.randn(fNum, nNum))).sum() / nNum / nNum

        return F.log_softmax(x, dim=1), 0, 0
