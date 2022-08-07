import torch
from torch.nn import Linear as Lin
import torch_geometric as tg
import torch.nn.functional as F
from torch import nn


class PGCN(torch.nn.Module):
    def __init__(self, input_dim, num_classes, dropout, hgc, lg, K = 3):
        super(PGCN, self).__init__()
        hidden = [hgc for i in range(lg)] 
        self.dropout = dropout
        bias = False
        self.relu = torch.nn.ReLU(inplace=True)
        self.lg = lg
        self.gconv = nn.ModuleList()
        for i in range(lg):
            in_channels = input_dim if i == 0 else hidden[i-1]
            # sym是采用对称归一化
            self.gconv.append(tg.nn.ChebConv(in_channels, hidden[i], K=K, normalization='sym', bias=bias))
        # cls_input_dim = sum(hidden)

        self.cls = nn.Sequential(
                # torch.nn.Linear(cls_input_dim, 256),
                torch.nn.Linear(hidden[lg-1], 256),
                torch.nn.ReLU(inplace=True),
                nn.BatchNorm1d(256), 
                torch.nn.Linear(256, num_classes))

        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, features, edge_index, edge_weight):

        x = self.relu(self.gconv[0](features, edge_index, edge_weight))
        # x0 = x
         
        for i in range(1, self.lg):
            x = F.dropout(x, self.dropout, self.training)
            x = self.relu(self.gconv[i](x, edge_index, edge_weight))
            # jk = torch.cat((x0, x), axis=1)
            # x0 = jk
        logit = self.cls(x)

        return logit, edge_weight

