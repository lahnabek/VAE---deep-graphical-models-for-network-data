import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- util ----------
def _gcn_norm(A_t: torch.Tensor) -> torch.Tensor:
    """Normalisation symm. GCN : G = D^{-1/2} (A+I) D^{-1/2}"""
    N = A_t.size(0)
    A_hat = A_t + torch.eye(N, device=A_t.device, dtype=A_t.dtype)
    deg = A_hat.sum(dim=1)
    D_inv_sqrt = torch.diag(torch.pow(torch.clamp(deg, min=1e-12), -0.5))
    return D_inv_sqrt @ A_hat @ D_inv_sqrt

# ---------- GCN ----------
class GCN(nn.Module):
    """
    - in_feats : dim des features d’entrée
    - hidden   : dim couche cachée
    - out_mu   : dim de sortie pour µ (ici Q-1)
    - out_lv   : dim de sortie pour logσ² (ici 1)
    GCN à 2 couches (Kipf-Welling) : H1 = ReLU(G X W0), µ = G H1 Wµ, logσ² = G H1 Wlv
    """
    def __init__(self, in_feats: int, hidden: int, out_mu: int, out_lv: int):
        super().__init__()
        self.W0   = nn.Linear(in_feats, hidden, bias=False)
        self.W_mu = nn.Linear(hidden, out_mu,  bias=False)
        self.W_lv = nn.Linear(hidden, out_lv,  bias=False)
        # init Xavier
        for m in [self.W0, self.W_mu, self.W_lv]:
            nn.init.xavier_uniform_(m.weight)

    def forward(self, A_t: torch.Tensor, X: torch.Tensor):
        G = _gcn_norm(A_t)          # N×N
        H1 = F.relu(G @ (X @ self.W0.weight.T))      # N×hidden
        mu =  G @ (H1 @ self.W_mu.weight.T)          # N×(Q-1)
        lv =  G @ (H1 @ self.W_lv.weight.T)          # N×1  (log-variance)
        return mu, lv
