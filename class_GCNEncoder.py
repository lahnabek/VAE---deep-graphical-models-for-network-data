import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNEncoder(torch.nn.Module):
    """
    Version PyTorch Geometric du GCN probabiliste.
    - in_feats : dimension des features d’entrée
    - hidden   : dimension de la couche cachée
    - out_mu   : dimension de sortie pour µ
    - out_lv   : dimension de sortie pour logσ²
    """
    def __init__(self, in_feats: int, hidden: int, out_mu: int, out_lv: int):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden)
        self.conv_mu = GCNConv(hidden, out_mu)
        self.conv_lv = GCNConv(hidden, out_lv)

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor):
        H = F.relu(self.conv1(X, edge_index))
        mu = self.conv_mu(H, edge_index)
        lv = self.conv_lv(H, edge_index)
        return mu, lv
