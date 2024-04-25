import torch
from torch_geometric.nn import GCNConv, ChebConv, GraphConv, global_mean_pool, BatchNorm
import torch.nn.functional as F
from torch.nn import Linear

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features, channels, num_classes, dropout):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.channels = channels
        self.conv1 = ChebConv(channels, hidden_channels,1)
        self.conv2 = ChebConv(hidden_channels, hidden_channels*2,2)
        self.conv3 = ChebConv(hidden_channels*2, hidden_channels*8,2)
        self.bn1 = BatchNorm(num_node_features)
        self.bn2 = BatchNorm(num_node_features)
        self.bn3 = BatchNorm(num_node_features)
        self.lin1 = Linear(hidden_channels*8, hidden_channels*2)
        self.lin2 = Linear(hidden_channels*2, hidden_channels)
        self.lin3 = Linear(hidden_channels, num_classes)
        self.soft = torch.nn.LogSoftmax(dim=-1)

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        # 1. Obtain node embeddings 
        x = x.squeeze(-1)
        #x = x.movedim(-1,-2)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = x.relu()
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        #
        # x = x.relu()
        # 2. Readout layer
        x = global_mean_pool(x, None)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin3(x)
        
        return self.soft(x)