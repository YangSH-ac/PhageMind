import torch
from torch.nn import Linear, LayerNorm, Sequential, ReLU
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
class GCNLinkPredictShare(torch.nn.Module):
    def __init__(self, in_dim0, in_dim1, hidden_dim, adapter_dim=None):
        super(GCNLinkPredictShare, self).__init__()
        self.indim0 = in_dim0
        self.hete0 = Linear(in_dim0, hidden_dim[0])
        self.hete1 = Linear(in_dim1, hidden_dim[0])
        self.conv1 = GCNConv(hidden_dim[0], hidden_dim[1])
        self.conv2 = GCNConv(hidden_dim[1], hidden_dim[2])
        self.lin0 = Linear(hidden_dim[2], hidden_dim[3])
        self.lin1 = Linear(hidden_dim[3], 1)
        if adapter_dim:
            self.adapter0 = Sequential(Linear(hidden_dim[0], adapter_dim), ReLU(), Linear(adapter_dim, hidden_dim[0]))
            self.adapter1 = Sequential(Linear(hidden_dim[0], adapter_dim), ReLU(), Linear(adapter_dim, hidden_dim[0]))
        self.adapter = True if adapter_dim else False
    def _check_size(self, data): 
        if data.edge_label.size(0) != 2 and data.edge_label.size(1) == 2:
            data.edge_label = data.edge_label.t()
        if data.edge_index.size(0) != 2 and data.edge_index.size(1) == 2:
            data.edge_index = data.edge_index.t()
    def heterodata(self, data):
        if data.small_idx == 1: 
            x0 = data.x * (1 - data.types).expand_as(data.x)
            x0 = x0[:, :self.indim0]
            x1 = data.x * data.types.expand_as(data.x)
            x0 = F.relu(self.hete0(x0))
            x1 = F.relu(self.hete1(x1))
        if data.small_idx == 2:
            x0 = data.x * data.types.expand_as(data.x)
            x1 = data.x * (1 - data.types).expand_as(data.x)
            x1 = x1[:, :self.indim0]
            x0 = F.relu(self.hete1(x0))
            x1 = F.relu(self.hete0(x1))
        return x0 + self.adapter0(x0) + x1 + self.adapter1(x1) if self.adapter else x0 + x1
    def forward(self, data):
        self._check_size(data)
        x = self.heterodata(data)
        x = F.relu(self.conv1(x,  data.edge_index))  
        x = F.relu(self.conv2(x,  data.edge_index))  
        x = F.relu(self.lin0(x))  
        return x
    def link_prediction(self, data):
        z = self.forward(data)
        z_u = z[data.edge_label[0]]
        z_v = z[data.edge_label[1]]
        out = torch.add(z_u, z_v)
        return self.lin1(out).squeeze()
class MLPLinkPredictShare(torch.nn.Module):
    def __init__(self, in_dim0, in_dim1, hidden_dim, adapter_dim=None):
        super(MLPLinkPredictShare, self).__init__()
        self.indim0 = in_dim0
        self.hete0 = Linear(in_dim0, hidden_dim[0])
        self.hete1 = Linear(in_dim1, hidden_dim[0])
        self.conv1 = Linear(hidden_dim[0], hidden_dim[1])
        self.conv2 = Linear(hidden_dim[1], hidden_dim[2])
        self.lin0 = Linear(hidden_dim[2], hidden_dim[3])
        self.lin1 = Linear(hidden_dim[3], 1)
        if adapter_dim:
            self.adapter0 = Sequential(Linear(hidden_dim[0], adapter_dim), ReLU(), Linear(adapter_dim, hidden_dim[0]))
            self.adapter1 = Sequential(Linear(hidden_dim[0], adapter_dim), ReLU(), Linear(adapter_dim, hidden_dim[0]))
        self.adapter = True if adapter_dim else False
    def _check_size(self, data): 
        if data.edge_label.size(0) != 2 and data.edge_label.size(1) == 2:
            data.edge_label = data.edge_label.t()
        if data.edge_index.size(0) != 2 and data.edge_index.size(1) == 2:
            data.edge_index = data.edge_index.t()
    def heterodata(self, data):
        if data.small_idx == 1:
            x0 = data.x * (1 - data.types).expand_as(data.x)
            x0 = x0[:, :self.indim0]
            x1 = data.x * data.types.expand_as(data.x)
            x0 = F.relu(self.hete0(x0))
            x1 = F.relu(self.hete1(x1))
        if data.small_idx == 2:
            x0 = data.x * data.types.expand_as(data.x)
            x1 = data.x * (1 - data.types).expand_as(data.x)
            x1 = x1[:, :self.indim0]
            x0 = F.relu(self.hete1(x0))
            x1 = F.relu(self.hete0(x1))
        return x0 + self.adapter0(x0) + x1 + self.adapter1(x1) if self.adapter else x0 + x1
    def forward(self, data):
        self._check_size(data)
        x = self.heterodata(data)
        x = F.relu(self.conv1(x))  
        x = F.relu(self.conv2(x))  
        x = F.relu(self.lin0(x))  
        return x
    def link_prediction(self, data):
        z = self.forward(data)
        z_u = z[data.edge_label[0]]
        z_v = z[data.edge_label[1]]
        out = torch.add(z_u, z_v)
        return self.lin1(out).squeeze()