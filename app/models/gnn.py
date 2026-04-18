import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch_geometric.nn import GATConv, global_mean_pool


class BugLocalizationGNN(torch.nn.Module):
    """
    3-layer Graph Attention Network for node-level bug classification.
    Input:  node feature vectors (embedding_dim + structural features)
    Output: 2-class logits per node (clean / buggy)
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()

        self.conv1 = GATConv(in_dim, hidden_dim, heads=4, dropout=0.3)
        self.bn1   = BatchNorm1d(hidden_dim * 4)

        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, dropout=0.3)
        self.bn2   = BatchNorm1d(hidden_dim * 4)

        self.conv3 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False, dropout=0.3)
        self.bn3   = BatchNorm1d(hidden_dim)

        self.classifier = Linear(hidden_dim, num_classes)
        self.dropout = 0.3

    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 3
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.elu(x)

        # Node-level classification
        return self.classifier(x)