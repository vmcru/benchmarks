import torch
from torch_geometric.nn import GCNConv, ChebConv, GraphConv, global_mean_pool
import torch.nn.functional as F
from torch.nn import Linear
import torch_geometric.nn.models as models
from torch_geometric.transforms.add_positional_encoding import AddLaplacianEigenvectorPE, AddRandomWalkPE



def from_static(x, edges):
    '''
    Take static graph and adapt it to useable graph for complex operations
    
    Arguments
    ---------
    x : torch.Tensor
        Input feature map
    edges : torch.tensor
        edge matrix
    '''
    from torch_geometric.data import Batch, Data
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
                dropout=0):
        super(GCN, self).__init__()

        if activation_type == "gelu":
            self.activation = torch.nn.GELU()
        elif activation_type == "elu":
            self.activation = torch.nn.ELU()
        elif activation_type == "relu":
            self.activation = torch.nn.ReLU()
        elif activation_type == "leaky_relu":
            self.activation = torch.nn.LeakyReLU()
        elif activation_type == "prelu":
            self.activation = torch.nn.PReLU()
        else:
            raise ValueError("Wrong hidden activation function")
        
        self.test = test

        if test==1:
            self.test = test
            self.model = models.GCN(in_channels=num_node_features, 
                                    hidden_channels=hidden_channels1, 
                                    out_channels=hidden_channels3,
                                    num_layers=num_layers,
                                    dropout=dropout,
                                    act=activation_type)
        elif test==2:
            self.test = test
            self.model = models.GraphSAGE(in_channels=num_node_features, 
                                        hidden_channels=hidden_channels1, 
                                        out_channels=hidden_channels3,
                                        num_layers=num_layers, 
                                        dropout=dropout,
                                        act=activation_type)
        elif test==3:
            self.text = test
            self.model = models.GAT(in_channels=num_node_features, 
                                    hidden_channels=hidden_channels1, 
                                    out_channels=hidden_channels3,
                                    num_layers=num_layers, 
                                    dropout=dropout,
                                    act=activation_type,
                                    heads = 2)
        elif test==4:
            self.text = test
            #TODO fix model setup
            self.model = models.PNA(in_channels=num_node_features, 
                                    hidden_channels=hidden_channels1, 
                                    out_channels=hidden_channels3,
                                    num_layers=num_layers, 
                                    dropout=dropout,
                                    act=activation_type,
                                    aggregators = 'var', 
                                    scalers = 'identity',
                                    deg = 2
                                    )
        else:    
            self.conv1 = ChebConv(num_node_features, hidden_channels1,kdeg1)
            self.conv2 = ChebConv(hidden_channels1, hidden_channels2,kdeg2)
            self.conv3 = ChebConv(hidden_channels2, hidden_channels3,kdeg3)
        self.lin = Linear(hidden_channels3, num_classes)
        self.soft = torch.nn.LogSoftmax(dim=1)

    def forward(self, batch):
        # 1. Obtain node embeddings 
        x, edge_index, _ = batch
        #x = x.squeeze(-1)
        x = x.movedim(-1,-2)
        edge_index = edge_index[0]
        if self.test!=0 :
            batch_new = from_static(x, edge_index)
            #AddLaplacianEigenvectorPE()
            
            x = self.model(batch_new.x, batch_new.edge_index.to(x.device))

        else:
            x = self.conv1(x, edge_index)
            x = self.activation(x)
            x = self.conv2(x, edge_index)
            x = self.activation(x)
            
            x = self.conv3(x, edge_index)
            x = self.activation(x)
            # 2. Readout layer

        x = global_mean_pool(x, batch_new.batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        if self.test==0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            x = x.squeeze()
        x = self.lin(x)
        return self.soft(x)
    

