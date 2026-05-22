import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT) for learning label correlations.
    Based on: https://arxiv.org/abs/1710.10903
    """
    
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # Learnable weight matrix W
        self.W = nn.Linear(in_features, out_features, bias=False)
        
        # Attention mechanism a
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, h, adj_matrix):
        """
        Args:
            h: Node feature matrix (N x in_features)
            adj_matrix: Adjacency matrix (N x N), can be dense or sparse
        
        Returns:
            h_prime: Updated node features (N x out_features)
        """
        # Linear transformation
        Wh = self.W(h)  # (N, out_features)
        N = Wh.size(0)
        
        # Compute attention coefficients
        # Concatenate every pair of nodes
        a_input = self._prepare_attentional_mechanism_input(Wh)  # (N, N, 2*out_features)
        
        # Apply attention function
        e = self.leakyrelu(self.a(a_input).squeeze(-1))  # (N, N)
        
        # Mask attention scores using adjacency matrix
        # Set -inf where there's no edge
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj_matrix > 0, e, zero_vec)
        
        # Normalize attention coefficients using softmax
        attention = F.softmax(attention, dim=1)  # (N, N)
        attention = self.dropout_layer(attention)
        
        # Aggregate neighbor features weighted by attention
        h_prime = torch.matmul(attention, Wh)  # (N, out_features)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
    
    def _prepare_attentional_mechanism_input(self, Wh):
        """
        Prepare pairwise concatenation of node features
        """
        N = Wh.size(0)
        
        # Repeat Wh along dimension 1: (N, 1, out_features) -> (N, N, out_features)
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)  # (N*N, out_features)
        
        # Repeat Wh along dimension 0: (1, N, out_features) -> (N, N, out_features)
        Wh_repeated_alternating = Wh.repeat(N, 1)  # (N*N, out_features)
        
        # Concatenate
        all_combinations_matrix = torch.cat(
            [Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1
        )  # (N*N, 2*out_features)
        
        return all_combinations_matrix.view(N, N, 2 * self.out_features)


class MultiHeadGATLayer(nn.Module):
    """
    Multi-head Graph Attention Layer
    """
    
    def __init__(self, in_features, out_features, num_heads, dropout=0.1, alpha=0.2, concat=True):
        super(MultiHeadGATLayer, self).__init__()
        self.num_heads = num_heads
        self.concat = concat
        
        # Create multiple attention heads
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(in_features, out_features, dropout, alpha, concat=True)
            for _ in range(num_heads)
        ])
        
        if concat:
            self.out_features = out_features * num_heads
        else:
            self.out_features = out_features
            
    def forward(self, h, adj_matrix):
        """
        Args:
            h: Node features (N x in_features)
            adj_matrix: Adjacency matrix (N x N)
        """
        if self.concat:
            # Concatenate outputs from all heads
            h_prime = torch.cat([att(h, adj_matrix) for att in self.attentions], dim=1)
        else:
            # Average outputs from all heads
            h_prime = torch.mean(torch.stack([att(h, adj_matrix) for att in self.attentions]), dim=0)
        
        return h_prime


class GAT(nn.Module):
    """
    Full GAT model with multiple layers
    """
    
    def __init__(self, in_features, hidden_features, out_features, 
                 num_heads=4, num_layers=2, dropout=0.1):
        super(GAT, self).__init__()
        
        self.num_layers = num_layers
        
        # First layer: multi-head attention
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(
            MultiHeadGATLayer(in_features, hidden_features, num_heads, dropout, concat=True)
        )
        
        # Middle layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                MultiHeadGATLayer(hidden_features * num_heads, hidden_features, 
                                 num_heads, dropout, concat=True)
            )
        
        # Output layer: single head, no concatenation
        if num_layers > 1:
            self.gat_layers.append(
                MultiHeadGATLayer(hidden_features * num_heads, out_features, 
                                 1, dropout, concat=False)
            )
        else:
            # If only one layer, output directly
            self.out_proj = nn.Linear(hidden_features * num_heads, out_features)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, adj_matrix):
        """
        Args:
            x: Node features (N x in_features)
            adj_matrix: Adjacency matrix (N x N)
        
        Returns:
            Updated node features (N x out_features)
        """
        for i, layer in enumerate(self.gat_layers):
            x = layer(x, adj_matrix)
            if i < len(self.gat_layers) - 1:
                x = self.dropout(x)
        
        return x
