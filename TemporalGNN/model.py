import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN



class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, num_timestamps_in, num_timestamps_out):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN(in_channels=node_features, 
                           out_channels=16, 
                           periods=num_timestamps_in)
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(16, node_features)        # x,y
        self.num_timestamps_in = num_timestamps_in
        self.num_timestamps_out = num_timestamps_out
        self.node_features = node_features

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        # predict auto-regressively
        output = []
        for i in range(self.num_timestamps_out):
            h = self.tgnn(x.double(), edge_index)
            h = F.relu(h)
            h = self.linear(h)
            h = h.reshape((h.shape[0], self.node_features, 1))
            x = torch.cat((x[:, :, 1:], h), dim=2)
            output.append(h)
        output = torch.cat(output, dim=2)
        return output