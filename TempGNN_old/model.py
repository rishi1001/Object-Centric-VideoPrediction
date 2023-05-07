import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN



class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, p, f):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN(in_channels=node_features, 
                           out_channels=16, 
                           periods=p)
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(16, f)

    def forward(self, x, edge_index,edge_weights):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = self.tgnn(x, edge_index,edge_weights)
        h = F.relu(h)
        h = self.linear(h)
        return h