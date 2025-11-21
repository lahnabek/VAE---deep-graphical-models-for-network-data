import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Normalisation GCN : G = D^{-1/2}(A+I)D^{-1/2} ----
def gcn_norm(A: torch.Tensor):
    N = A.size(0)
    A_hat = A + torch.eye(N, device=A.device, dtype=A.dtype)
    deg = A_hat.sum(dim=1)
    D_inv_sqrt = torch.diag(torch.pow(torch.clamp(deg, 1e-12), -0.5))
    return D_inv_sqrt @ A_hat @ D_inv_sqrt


class GCNMultiLayers(nn.Module):
    """
    GCN flexible :
      - num_layers >= 1
      - hidden : liste de dimensions [H1, H2, ..., HL]
      - split_last_layer :
            False → µ et logσ² partent du même H_L
            True  → 2 sous-couches GCN dans la dernière couche
    """
    def __init__(self, 
                 in_feats: int,
                 hidden: list,
                 out_mu: int,
                 out_lv: int,
                 split_last_layer: bool = False):

        super().__init__()

        assert len(hidden) >= 1, "hidden_dims doit contenir au moins une dimension."
        self.num_layers = len(hidden)
        self.split_last_layer = split_last_layer

        layers = []
        input_dim = in_feats

        # ---- 1) Construction des couches cachées ----
        for h in hidden:
            W = nn.Linear(input_dim, h, bias=False)
            nn.init.xavier_uniform_(W.weight)
            layers.append(W)
            input_dim = h

        self.layers = nn.ModuleList(layers)

        # ---- 2) Couches finales (µ, logσ²) ----
        if not split_last_layer:
            # Une seule dernière couche -> on partage H_L
            self.W_mu = nn.Linear(hidden[-1], out_mu, bias=False)
            self.W_lv = nn.Linear(hidden[-1], out_lv, bias=False)
            nn.init.xavier_uniform_(self.W_mu.weight)
            nn.init.xavier_uniform_(self.W_lv.weight)

        else:
            # Dernière couche séparée GCN → H_L -> µ  et  H_L -> logσ²
            H_last = hidden[-1]
            self.W_mu = nn.Linear(H_last, out_mu, bias=False)
            self.W_lv = nn.Linear(H_last, out_lv, bias=False)

            nn.init.xavier_uniform_(self.W_mu.weight)
            nn.init.xavier_uniform_(self.W_lv.weight)

    def forward(self, A: torch.Tensor, X: torch.Tensor):
        G = gcn_norm(A)

        H = X
        
        # ---- Passages successifs dans les couches GCN ----
        for i, W in enumerate(self.layers):            
            H = G @ (H @ W.weight.T)
            # On met ReLU sauf pour la dernière couche si split_last_layer=True
            if i < self.num_layers - 1:
                H = F.relu(H)

        # ---- Couches finales ----
        mu = G @ (H @ self.W_mu.weight.T)
        lv = G @ (H @ self.W_lv.weight.T)

        return mu, lv
