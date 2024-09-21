import torch
from torch_geometric.nn import GCNConv, ChebConv, GraphConv, global_mean_pool
import torch.nn.functional as F
from torch.nn import Linear
import torch_geometric.nn.models as models
from torch_geometric.data import Batch, Data

def from_static(x, edges):
    '''
    Take static graph and adapt it to usable graph for complex operations.
    
    Arguments
    ---------
    x : torch.Tensor
        Input feature map.
    edges : torch.tensor
        Edge matrix.
    '''
    if len(edges.shape) == 3:
        data_list = [Data(x=x_, edge_index=edges[0]) for x_ in x]
    else:
        data_list = [Data(x=x_, edge_index=edges) for x_ in x] 
    batch = Batch.from_data_list(data_list) 
    return batch

class GCN(torch.nn.Module):
    def __init__(self, 
                num_classes, 
                channels, 
                kdeg1=1, 
                kdeg2=1, 
                kdeg3=1, 
                hidden_channels1=16, 
                hidden_channels2=16, 
                hidden_channels3=16, 
                num_node_features=16, 
                activation_type="elu", 
                test=0, 
                num_layers=1, 
                dropout=0,
                heads=1):
        super(GCN, self).__init__()

        # Choose activation function
        activations = {
            "gelu": torch.nn.GELU(),
            "elu": torch.nn.ELU(),
            "relu": torch.nn.ReLU(),
            "leaky_relu": torch.nn.LeakyReLU(),
            "prelu": torch.nn.PReLU(),
        }
        self.activation = activations.get(activation_type)
        if self.activation is None:
            raise ValueError(f"Unsupported activation function: {activation_type}")

        self.test = test

        # Initialize the appropriate model based on the `test` flag
        if test == 1:
            self.model = models.GCN(in_channels=num_node_features, 
                                    hidden_channels=hidden_channels1, 
                                    out_channels=hidden_channels3,
                                    num_layers=num_layers,
                                    dropout=dropout,
                                    act=activation_type)
        elif test == 2:
            self.model = models.GraphSAGE(in_channels=num_node_features, 
                                        hidden_channels=hidden_channels1, 
                                        out_channels=hidden_channels3,
                                        num_layers=num_layers, 
                                        dropout=dropout,
                                        act=activation_type)
        elif test == 3:
            self.model = models.GAT(in_channels=num_node_features, 
                                    hidden_channels=hidden_channels1, 
                                    out_channels=hidden_channels3,
                                    num_layers=num_layers, 
                                    dropout=dropout,
                                    act=activation_type,
                                    heads=heads)
        elif test == 4:
            self.model = models.PNA(in_channels=num_node_features, 
                                    hidden_channels=hidden_channels1, 
                                    out_channels=hidden_channels3,
                                    num_layers=num_layers, 
                                    dropout=dropout,
                                    act=activation_type,
                                    aggregators=['mean', 'var', 'min', 'max'], 
                                    scalers=['identity', 'amplification', 'attenuation'],
                                    deg=torch.tensor([2]))  # Assuming `deg` is a tensor or replace with actual degree tensor
        else:
            self.conv1 = ChebConv(num_node_features, hidden_channels1, kdeg1)
            self.conv2 = ChebConv(hidden_channels1, hidden_channels2, kdeg2)
            self.conv3 = ChebConv(hidden_channels2, hidden_channels3, kdeg3)
        
        self.lin = Linear(hidden_channels3, num_classes)
        self.dropout = dropout
        self.soft = torch.nn.LogSoftmax(dim=1)

    def forward(self, batch):
        # Extract the necessary components from the batch
        x,_ ,edge_index  = batch

        x = x.movedim(-1,-2)
        
        newbatch = from_static(x,edge_index)

        if self.test != 0:
            # If using one of the predefined models
            x = self.model(newbatch.x, newbatch.edge_index.to(x.device))
        else:
            # Custom graph processing with ChebConv layers
            x = self.conv1(newbatch.x, newbatch.edge_index.to(x.device))
            x = self.activation(x)
            x = self.conv2(x, newbatch.edge_index.to(x.device))
            x = self.activation(x)
            x = self.conv3(x, newbatch.edge_index.to(x.device))
            x = self.activation(x)
        
        # Global mean pooling over nodes
        x = global_mean_pool(x, newbatch.batch.to(x.device))  # [batch_size, hidden_channels]

        # Apply dropout
        if self.test == 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final classifier layer
        x = self.lin(x)
        return self.soft(x)
