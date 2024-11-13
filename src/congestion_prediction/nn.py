import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm
from torch_geometric.data import Data

"""
The NN architectures for both the congestion and flow prediction models. 
"""

class MAPFCongestionModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_gnn_layers: int, 
                 num_attention_heads: int, max_time_steps: int, dropout_rate: float = 0.2):
        super().__init__()
        self.max_time_steps = max_time_steps
        self.dropout_rate = dropout_rate

        # GNN layers with batch normalization
        self.gnn_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            self.gnn_layers.append(GCNConv(input_dim if i == 0 else hidden_dim, hidden_dim))
            self.bn_layers.append(BatchNorm(hidden_dim))

        # GRU layer with dropout
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.gru_dropout = nn.Dropout(dropout_rate)

        # Multi-head attention with layer normalization
        self.attention = nn.MultiheadAttention(hidden_dim, num_attention_heads, dropout=dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Output layer with dropout
        self.output_layer = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        batch_size = x.size(0) // self.max_time_steps
        num_nodes = x.size(1)
        feature_dim = x.size(2)
        x = x.view(batch_size, self.max_time_steps, num_nodes, feature_dim)

        x = self._apply_gnn(x, edge_index, batch_size, num_nodes, feature_dim)
        x = self._apply_gru(x, batch_size, num_nodes)
        x = self._apply_attention(x, batch_size, num_nodes)
        output = self._get_output(x, batch_size, self.max_time_steps, num_nodes)

        return output

    def _apply_gnn(self, x: torch.Tensor, edge_index: torch.Tensor,
               batch_size: int, num_nodes: int, feature_dim: int) -> torch.Tensor:
        outputs = []
        for t in range(self.max_time_steps):
            x_t = x[:, t].reshape(batch_size * num_nodes, feature_dim)
            for gnn_layer, bn_layer in zip(self.gnn_layers, self.bn_layers):
                x_t = gnn_layer(x_t, edge_index)
                x_t = bn_layer(x_t)
                x_t = F.relu(x_t)
                x_t = F.dropout(x_t, p=self.dropout_rate, training=self.training)
            outputs.append(x_t.view(batch_size, num_nodes, -1))
        return torch.stack(outputs, dim=1)

    def _apply_gru(self, x: torch.Tensor, batch_size: int, num_nodes: int) -> torch.Tensor:
        x = x.view(batch_size * num_nodes, self.max_time_steps, -1)
        x, _ = self.gru(x)
        x = self.gru_dropout(x)
        return x.view(batch_size, num_nodes, self.max_time_steps, -1)

    def _apply_attention(self, x: torch.Tensor, batch_size: int, num_nodes: int) -> torch.Tensor:
        x = x.permute(2, 0, 1, 3).contiguous()
        x = x.view(self.max_time_steps, batch_size * num_nodes, -1)
        
        # Apply attention mechanism
        attn_output, _ = self.attention(x, x, x)
        
        # Residual connection and layer normalization
        x = x + attn_output
        x = self.layer_norm(x)
        
        return x.view(self.max_time_steps, batch_size, num_nodes, -1).permute(1, 2, 0, 3).contiguous()

    def _get_output(self, x: torch.Tensor, batch_size: int,
                    max_time_steps: int, num_nodes: int) -> torch.Tensor:
        return self.output_layer(x).view(batch_size, max_time_steps, num_nodes)
    




