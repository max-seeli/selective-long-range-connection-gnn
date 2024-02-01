import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

from common import LAST_LAYER

class QM9GNN(nn.Module):
    def __init__(self, gnn_type, num_layers, dim0, h_dim, out_dim, last_layer,
                 unroll, layer_norm, use_activation, use_residual):
        super(QM9GNN, self).__init__()
        self.gnn_type = gnn_type
        self.unroll = unroll
        self.last_layer = last_layer
        self.use_layer_norm = layer_norm
        self.use_activation = use_activation
        self.use_residual = use_residual
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.feature_embed = nn.Linear(dim0, h_dim)

        self.num_layers = num_layers        
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        if unroll:
            self.layers.append(gnn_type.get_layer(
                in_dim=h_dim,
                out_dim=h_dim))
        else:
            for i in range(num_layers):
                self.layers.append(gnn_type.get_layer(
                    in_dim=h_dim,
                    out_dim=h_dim))
        if self.use_layer_norm:
            for i in range(num_layers):
                self.layer_norms.append(nn.LayerNorm(h_dim))

        self.out_dim = out_dim
        self.out_layer = nn.Linear(in_features=h_dim, out_features=out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.feature_embed(x)
        x = F.relu(x)

        for i in range(self.num_layers):
            if self.unroll:
                layer = self.layers[0]
            else:
                layer = self.layers[i]
            new_x = x
            
            if self.last_layer == LAST_LAYER.FULLY_ADJACENT and i == self.num_layers - 1:
                num_nodes = data.num_nodes
                edges = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
                edges = torch.tensor(edges).t().to(self.device)
            elif self.last_layer == LAST_LAYER.K_HOP and i == self.num_layers - 1:
                k_hop_edge_index = data.k_hop_edge_index
                edges = k_hop_edge_index
            else:
                edges = edge_index

            x = layer(x, edges)

            if self.use_layer_norm:
                x = self.layer_norms[i](x)
            if self.use_activation:
                x = F.relu(x)
            if self.use_residual:
                x = x + new_x
        
        x = gnn.global_mean_pool(x, batch)
        x = self.out_layer(x)
        return x.squeeze(-1)
