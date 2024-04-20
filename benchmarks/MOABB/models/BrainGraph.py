import torch
from torch_geometric.nn import GCNConv, ChebConv, GraphConv, global_mean_pool
import torch.nn.functional as F
from torch.nn import Linear

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features, num_classes, dropout):
        super(GCN, self).__init__()
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.conv1 = ChebConv(num_node_features, hidden_channels,1)
        self.conv2 = ChebConv(hidden_channels, hidden_channels,1)
        self.conv3 = ChebConv(hidden_channels, hidden_channels,1)
        self.lin = Linear(hidden_channels, num_classes)
        self.soft = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x, edge_index):
        # 1. Obtain node embeddings 
        x = x.squeeze(-1)
        #x = x.movedim(-1,-2)
        edge_index = edge_index[0]
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        
        x = self.conv3(x, edge_index)
        
        # 2. Readout layer
        x = global_mean_pool(x, None)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        
        return self.soft(x)