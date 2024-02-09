
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear5(nn.Module):
    def __init__(self, in_feature=784, hidden_feature=784,n_class = 10, bias=False):
        super(Linear5, self).__init__()
        self.fc1 = nn.Linear(in_feature, hidden_feature, bias=bias)
        self.fc2 = nn.Linear(hidden_feature, hidden_feature, bias=bias)
        self.fc3 = nn.Linear(hidden_feature, hidden_feature, bias=bias)
        self.fc4 = nn.Linear(hidden_feature, hidden_feature, bias=bias)
        self.fc5 = nn.Linear(hidden_feature, n_class, bias=bias)

        # stores the activations for gpm
        # self.act=OrderedDict()

    def forward(self, x):        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.dropout(F.relu(x),p=.1)
        x = self.fc2(x)
        x = F.dropout(F.relu(x),p=.1)
        x = self.fc3(x)
        x = F.dropout(F.relu(x),p=.1)
        x = self.fc4(x)
        x = F.dropout(F.relu(x),p=.1)
        x = self.fc5(x)
        return x