'''
Implementation de l'algorithme deepLPBM, qui est la fonction main().
Main applique deep_LPBM à fichier numero SUBJECT_IDX dans le repertoire DATA_DIR
MODE est outile seulement dans le cas de données synthetiques, on peut poser MODE="" sinon.
Il sauvegarde tout résultat dans le repertoire "results+MODE"
'''

import os, json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix, adjusted_rand_score, normalized_mutual_info_score
from torch_geometric.utils import dense_to_sparse
import seaborn as sns
import networkx as nx


import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Patch
import matplotlib.colors as mcolors
import time



from class_GCN import *
from class_GCNEncoder import *
from class_GCN_Multi_layers import *
# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------


MODE = "assortative" # disassortative, assortative, hub
DATA_DIR = "data_synthetic/" + MODE# "miscdata" 
RANDOM_STATE = 42
HIDDEN_LAYERS_GCN = [ 32]
SPLITLAYER = False
MAX_EPOCHS_INIT = 300     # Phase init encodeur (Algorithme 1)
MAX_EPOCHS = 600          # Phase estimation (Algorithme 1)
LEARNING_RATE = 1e-2   # Adam lr=0.01 dans l'article
NUM_SEEDS = 5            # on garde le meilleur ELBO
CLASS_GCN = "GCNEncoder"  #  "GCNEncoder" , 
SUBJECT_IDX = 0
METRIC = "AIC" # AIC, BIC, ICL
Q_list = [3, 4, 5, 6, 7]
NEG_RATIO = 5
USE_NEGATIVE_SAMPLING = False

config_zero = {
    'MODE' : MODE, # disassortative, assortative, hub
    'DATA_DIR' : DATA_DIR, # "miscdata" 
    'RANDOM_STATE' : RANDOM_STATE,
    'HIDDEN_LAYERS_GCN' : HIDDEN_LAYERS_GCN, 
    'SPLITLAYER' : SPLITLAYER,
    'MAX_EPOCHS_INIT' : MAX_EPOCHS_INIT,     # Phase init encodeur (Algorithme 1)
    'MAX_EPOCHS' :MAX_EPOCHS,          # Phase estimation (Algorithme 1)
    'LEARNING_RATE' : LEARNING_RATE,   # Adam lr=0.01 dans l'article
    'NUM_SEEDS' : NUM_SEEDS,           # on garde le meilleur ELBO
    'CLASS_GCN' : CLASS_GCN,  #  "GCNEncoder" , 
    'SUBJECT_IDX' : SUBJECT_IDX,
    'METRIC' : "AIC",
    'Q_list' : Q_list,
    'NEG_RATIO': NEG_RATIO,
    'USE_NEGATIVE_SAMPLING' : USE_NEGATIVE_SAMPLING  }

import matplotlib
"""
ancienne config GCN 2 couches de l'article, marche bien sur Q=3 mais pas au dela:
HIDDEN_LAYERS_GCN = [32]
SPLITLAYER = False
"""
# ---------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------


def init_all_params(A, Q, params, eps=1e-6, tau=1e-16):
    """
    Initialise TOUT pour Deep LPBM (VERSION DOUCE, type ancien code):

      - eta0       : N×Q appartenance douce issue de KMeans (one-hot + tau)
      - z0         : N×(Q-1) logits initiaux (log((C+tau)/(C_last+tau)))
      - Pi0        : Q×Q estimation initiale des probabilités de blocs (adoucie)
      - Pi_tilde0  : Q×Q paramètre non borné (nn.Parameter), stabilisé

    Version empirique STABLE, proche de ton comportement original → ARI élevé.
    """

    device = torch.device(params.get("device", "cpu"))
    N = A.shape[0]
    A_t = torch.tensor(A, dtype=torch.float32, device=device)

    # ===========================
    # 1) KMEANS → labels
    # ===========================
    km = KMeans(n_clusters=Q, n_init="auto", random_state=0).fit(A)
    labels = torch.from_numpy(km.labels_.astype(int)).to(device)

    # ===========================
    # 2) One-hot dur → ADOUCISSEMENT "ancien code"
    # ===========================
    C = F.one_hot(labels, num_classes=Q).float().to(device)  # shape (N,Q)

    # adoucir légèrement (évite extrêmes, mais garde forme one-hot)
    eta0 = C * (1 - (Q-1)*tau) + (1 - C) * tau
    eta0 = eta0 / eta0.sum(dim=1, keepdim=True)  # normalisation simplex

    # ===========================
    # 3) z0 = LOG-RATIOS (ancienne méthode) + stabilisation
    # ===========================
    
    denom = eta0[:, [Q-1]] + tau
    numer = eta0[:, :Q-1] + tau
    z0 = torch.log(numer) - torch.log(denom)

    # clamp modéré pour éviter z explosifs
    z0 = torch.clamp(z0, -8.0, 8.0) # excessif avec toutes les protections

    # ===========================
    # 4) Estimation douce de Pi0 (SBM-like)
    # ===========================
    Pi0_num = torch.zeros((Q, Q), device=device)
    Pi0_den = torch.zeros((Q, Q), device=device)
    upper_mask = torch.triu(torch.ones(N, N, device=device), diagonal=1)

    for q in range(Q):
        eta_q = eta0[:, q].unsqueeze(1)      # (N,1)
        for r in range(Q):
            eta_r = eta0[:, r].unsqueeze(0)  # (1,N)
            eta_outer = eta_q @ eta_r        # (N,N)

            num = (A_t * eta_outer * upper_mask).sum()
            den = (eta_outer * upper_mask).sum()

            Pi0_num[q, r] = num
            Pi0_den[q, r] = den + eps

    Pi0 = Pi0_num / Pi0_den
    Pi0 = 0.5 * (Pi0 + Pi0.T)

    # clamp léger : Pi0 jamais trop extrême
    Pi0 = Pi0.clamp(0.05, 0.95)

    # ===========================
    # 5) Pi_tilde0 = inverse arctan (stabilisé)
    # ===========================
    Pi_tilde0 = torch.tan(np.pi * (Pi0 - 0.5))

    # clamp Pĩ0 pour éviter explosions au tout début
    #Pi_tilde0 = torch.clamp(Pi_tilde0, -8.0, 8.0)

    # convertir en nn.Parameter directement ici
    Pi_tilde0 = nn.Parameter(Pi_tilde0)

    # ===========================
    # Stockage dans params
    # ===========================
    params["eta0"]       = eta0
    params["z0"]         = z0
    params["Pi0"]        = Pi0
    params["Pi_tilde0"]  = Pi_tilde0

    return eta0, z0, Pi0, Pi_tilde0


def softmax_logits_to_eta(z):
    """
    - z : N×(Q−1) logits per node (last coord has implicit 0-logit)
    Returns η (N×Q) row-stochastic partial memberships; stable softmax via row-wise max.
    Article: Section 2 (logistic-normal with Q−1 logits).
    """
    eps = 1e-12 # ne pas diviser par 0
    N = z.size(0)
    z_pad = torch.cat([z, torch.zeros((N, 1), device=z.device, dtype=z.dtype)], dim=1)
    m = torch.amax(z_pad, dim=1, keepdim=True)                # max ligne
    e = torch.exp(z_pad - m)
    eta = e / (e.sum(dim=1, keepdim=True) + eps)
    return eta

def reparameterize(mu, logvar):
    """
    Reparametrization trick : z = mu + sigma * eps.
    Rappel article : Sec. 4.2 (optimisation stochastique).
    """
    eps = torch.randn_like(mu)
    sigma = torch.exp(0.5 * logvar).expand_as(mu)
    return mu + sigma * eps




def unconstrained_Pi_to_Pi(Pi_tilde):
    """
    f: R -> (0,1) élément par élément pour obtenir Π.
    Rappel article : mapping bijectif pour optimiser Π~ sans contraintes (Sec. 4.2).
    f(x) = 0.5 + (1/pi) * arctan(x)
    """
    Pi = 0.5 + torch.atan(Pi_tilde) / np.pi
    return 0.5 * (Pi + Pi.T) # symmétrie

def init_encoder_phase(A, Q, params, config, results_dir=None):
    """
    Phase d'initialisation de l’encodeur GCN (Algorithme 1 / App. B.2).
    - µ ≈ z₀ (issu de KMeans)
    - logσ² ≈ log(0.01)

    Sauvegarde (dans results_dir) :
        - répartition initiale KMeans (CSV + histogramme)
        - courbe de perte d'initialisation init_loss.png
    """
    device   = torch.device(params.get("device", "cpu"))
    hidden   = (params.get("hidden", config['HIDDEN_LAYERS_GCN']))
    init_lr  = float(params.get("init_lr", config['LEARNING_RATE']))
    seed     = int(params.get("seed", config['RANDOM_STATE']))
    tau      = float(params.get("init_tau", 1e-3))

    torch.manual_seed(seed)
    N = A.shape[0]

    # --- Création du dossier (si demandé) ---
    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)

    # --- Modèle GCN (créé une fois et stocké dans params)
    #gcn = GCN(in_feats=N, hidden=hidden, out_mu=Q-1, out_lv=1).to(device)
    gcn = GCNMultiLayers(in_feats=N, hidden=hidden, out_mu=Q-1, out_lv=1, split_last_layer=config['SPLITLAYER']).to(device)
    

    # --- Optimiseur pour la phase d'init
    opt = torch.optim.Adam(gcn.parameters(), lr=init_lr)


    # --- Tenseurs d’entrée
    A_t = torch.tensor(A, dtype=torch.float32, device=device)
    X   = torch.eye(N, dtype=torch.float32, device=device)

    # --- Initialisation douce via KMeans ---
    eta0, z0, Pi0, Pi_tilde0 = init_all_params(A, Q, params)
    lv_star = torch.full((N, 1), np.log(0.01), dtype=torch.float32, device=device)

    # --- Boucle d’entraînement (phase d’init) ---
    loss_history = []
    gcn.train()
    for _ in range(config['MAX_EPOCHS_INIT']):
        opt.zero_grad()
        mu, logvar = gcn(A_t, X)
        loss_mu = F.mse_loss(mu, z0)
        loss_lv = F.mse_loss(logvar, lv_star)
        loss = loss_mu + loss_lv
        loss.backward()
        opt.step()
        loss_history.append(float(loss.detach().cpu()))

    last_loss = loss_history[-1] if loss_history else None
    params["gcn"] = gcn
    # --- Sauvegarde de la courbe de loss ---
    if results_dir is not None:
        plt.figure()
        plt.plot(loss_history)
        plt.xlabel("Itérations")
        plt.ylabel("Loss (µ, logσ²)")
        plt.title(f"Phase d'initialisation GCN (Q={Q})")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "init_loss.png"))
        plt.close()

    return last_loss


def init_encoder_phase_GCNEncoder(A, Q, params, config, results_dir=None):
    """
    Phase d'initialisation de l’encodeur GCN (version PyTorch Geometric).

    - µ ≈ z₀ (issu de KMeans sur A)
    - logσ² ≈ log(0.01)
    - Utilise la nouvelle classe GCN basée sur GCNConv
      (entrée : X, edge_index)

    Sauvegarde dans results_dir :
        - répartition initiale KMeans (CSV + histogramme)
        - courbe de perte d'initialisation init_loss.png
    """
    device   = torch.device(params.get("device", "cpu"))
    hidden   = int(params.get("hidden", config['HIDDEN_LAYERS_GCN'][0]))
    init_lr  = float(params.get("init_lr", config['LEARNING_RATE']))
    seed     = int(params.get("seed", config['RANDOM_STATE']))
    tau      = float(params.get("init_tau", 1e-3))

    torch.manual_seed(seed)
    N = A.shape[0]

    # --- Création du dossier (si demandé) ---
    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)

    # --- Conversion adjacency → edge_index ---
    A_t = torch.tensor(A, dtype=torch.float32, device=device)
    edge_index, _ = dense_to_sparse(A_t)

    # --- Features d’entrée : ici identité
    X = torch.eye(N, dtype=torch.float32, device=device)

    # --- Modèle GCN (créé une seule fois et stocké dans params)
    gcn = GCNEncoder(in_feats=N, hidden=hidden, out_mu=Q-1, out_lv=1).to(device)
    

    # --- Optimiseur pour la phase d'init
    opt = torch.optim.Adam(gcn.parameters(), lr=init_lr)
    

    # --- Initialisation douce via KMeans ---
    eta0, z0, Pi0, Pi_tilde0 = init_all_params(A, Q, params)
    # Sauvegarde debug
    save_init_debug_figures(A, eta0, z0, Pi0, Pi_tilde0, results_dir)

    lv_star = torch.full((N, 1), np.log(0.01), dtype=torch.float32, device=device)

    # --- Boucle d’entraînement (phase d’init) ---
    loss_history = []
    gcn.train()
    for _ in range(config['MAX_EPOCHS_INIT']):
        opt.zero_grad()

        # passage GCNConv → µ, logσ²
        mu, logvar = gcn(X, edge_index)

        # alignement sur z₀ et logσ² cible
        loss_mu = F.mse_loss(mu, z0)
        loss_lv = F.mse_loss(logvar, lv_star)
        loss = loss_mu + loss_lv
        loss.backward()
        opt.step()

        loss_history.append(float(loss.detach().cpu()))

    last_loss = loss_history[-1] if loss_history else None

    params["gcn"] = gcn
    # --- Sauvegarde de la courbe de loss ---
    if results_dir is not None:
        plt.figure()
        plt.plot(loss_history)
        plt.xlabel("Itérations")
        plt.ylabel("Loss (µ, logσ²)")
        plt.title(f"Phase d'initialisation GCN (Q={Q})")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "init_loss.png"))
        plt.close()

    return last_loss


def decoder_prob(Z, Pi, eps=1e-8):
    """
    - Z  : N×(Q−1) logits (échantillon)
    - Pi : Q×Q (matrice de blocs)
    Calcule P (N×N) avec p_ij = η_i^T Π η_j (Sec. 2 → η, Sec. 4 → décodeur bloc).
    """
    # Z → η (forme Sec. 2 : on ajoute le logit 0 implicite puis softmax stable)
    # NB: softmax_logits_to_eta attend Z de taille N×(Q−1) et gère le padding interne.
    eta = softmax_logits_to_eta(Z)                # N×Q
    P = eta @ Pi @ eta.T                          # N×N
    P = torch.clamp(P, eps, 1 - eps)                  # stabilité num.
    return P


# -----------------------------------------------------------
# DEBUG INIT
# -----------------------------------------------------------
def make_debug_dir(results_dir, name="init_debug"):
    debug_dir = os.path.join(results_dir, name)
    os.makedirs(debug_dir, exist_ok=True)
    return debug_dir



def save_init_debug_figures(A, eta0, z0, Pi0, Pi_tilde0, results_dir):
    if results_dir is None:
        return
    debug_dir = make_debug_dir(results_dir)

    # --- FIGURE 1 : Heatmap de eta0 ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(eta0.cpu().numpy(), cmap="viridis")
    plt.title("η₀ (initial soft memberships)")
    plt.xlabel("Cluster q")
    plt.ylabel("Node i")
    plt.tight_layout()
    plt.savefig(os.path.join(debug_dir, "eta0_heatmap.png"))
    plt.close()

    # --- FIGURE 2 : Heatmap de z0 ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(z0.cpu().numpy(), cmap="coolwarm", center=0)
    plt.title("z₀ (initial logits)")
    plt.xlabel("Dim q")
    plt.ylabel("Node i")
    plt.tight_layout()
    plt.savefig(os.path.join(debug_dir, "z0_heatmap.png"))
    plt.close()

    # --- FIGURE 3 : Histogramme des z0 ---
    plt.figure(figsize=(6, 5))
    plt.hist(z0.cpu().numpy().flatten(), bins=40)
    plt.title("Distribution de z₀")
    plt.tight_layout()
    plt.savefig(os.path.join(debug_dir, "z0_histogram.png"))
    plt.close()

    # --- FIGURE 4 : Heatmap de Pi0 ---
    plt.figure(figsize=(6, 5))
    sns.heatmap(Pi0.cpu().numpy(), annot=True, fmt=".2f", cmap="viridis")
    plt.title("Π₀ (initial block matrix)")
    plt.xlabel("Cluster r")
    plt.ylabel("Cluster q")
    plt.tight_layout()
    plt.savefig(os.path.join(debug_dir, "Pi0_heatmap.png"))
    plt.close()

    # --- FIGURE 5 : Heatmap de Pĩ0 ---
    plt.figure(figsize=(6, 5))
    sns.heatmap(Pi_tilde0.detach().cpu().numpy(), cmap="coolwarm", center=0)
    plt.title("Π̃₀ (initial unconstrained block matrix)")
    plt.tight_layout()
    plt.savefig(os.path.join(debug_dir, "Pi_tilde0_heatmap.png"))
    plt.close()

    # --- FIGURE 6 : Histogramme KMeans init ---
    labels = eta0.argmax(dim=1).cpu().numpy()
    plt.figure(figsize=(5, 4))
    plt.hist(labels, bins=np.arange(eta0.shape[1]+1)-0.5, rwidth=0.8)
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.title("Histogramme des clusters KMeans initiaux")
    plt.tight_layout()
    plt.savefig(os.path.join(debug_dir, "kmeans_hist.png"))
    plt.close()

    

# -----------------------------------------------------------
# REPRESENTATION
# -----------------------------------------------------------
def plot_elbo_curve(elbo_history, Q, save_dir):
    """Affiche et enregistre la courbe ELBO."""
    if elbo_history is None or len(elbo_history) == 0:
        return
    plt.figure()
    plt.plot(elbo_history)
    plt.xlabel("Itérations")
    plt.ylabel("ELBO")
    plt.title(f"Évolution de l'ELBO (Q={Q})")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"ELBO_Q{Q}.png"))
    plt.close()

def plot_Pi_matrix(Pi, Q, save_dir):
    """Affiche et enregistre la matrice Π finale."""
    plt.figure()
    plt.imshow(Pi, cmap="viridis")
    plt.colorbar(label="Probabilité de connexion")
    plt.title(f"Matrice Π finale (Q={Q})")
    plt.xlabel("Cluster j")
    plt.ylabel("Cluster i")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"Pi_Q{Q}.png"))
    plt.close()

def plot_eta_histogram(eta, Q, save_dir):
    """Affiche et enregistre l’histogramme des clusters (argmax η)."""
    plt.figure()
    plt.hist(np.argmax(eta, axis=1), bins=np.arange(Q + 1) - 0.5, rwidth=0.8)
    plt.xlabel("Cluster assigné (argmax η)")
    plt.ylabel("Nombre de nœuds")
    plt.title(f"Distribution des appartenances (Q={Q})")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"eta_hist_Q{Q}.png"))
    plt.close()

def save_all_figures(results_dir, best, Q):
    """Fonction principale pour enregistrer toutes les figures standard."""
    os.makedirs(results_dir, exist_ok=True)
    plot_elbo_curve(best.get("history"), Q, results_dir)
    plot_Pi_matrix(best["Pi"], Q, results_dir)
    plot_eta_histogram(best["eta"], Q, results_dir)


def draw_graph_with_probabilities(A, eta, class_colors=None, class_labels=None, node_radius=None, results_dir=None, show=False, title=None, add_title = ""):
    """
    A : (n,n) 
    eta : (n,k) 
    class_colors : coleurs, facultatives
    node_radius : taille des nodes
    """
    n, k = eta.shape

    if not node_radius:
        node_radius = 4/n


    if not class_colors:
        cmap = plt.get_cmap("tab10")
        class_colors = [cmap(i % cmap.N) for i in range(k)]

    
    if class_labels is None:
        class_labels = [f"Class {i}" for i in range(k)]


    G = nx.from_numpy_array(A)
    pos = nx.spring_layout(G, seed=0)

    fig, ax = plt.subplots(figsize=(8,8))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4, edge_color='grey')
    
    for i, (x, y) in pos.items():
        start_angle = 90
        for frac, color in zip(eta[i], class_colors):
            if frac <= 0:  # skip empty slices
                continue
            theta1 = start_angle
            theta2 = start_angle + 360 * frac
            wedge = Wedge(center=(x, y),
                          r=node_radius,
                          theta1=theta1,
                          theta2=theta2,
                          facecolor=color,
                          edgecolor='black',
                          linewidth=0.3
                          )
            ax.add_patch(wedge)
            start_angle = theta2
    
    for (x, y) in pos.values():
        circ = plt.Circle((x, y), node_radius, fill=False, edgecolor='black', lw=0.3)
        ax.add_patch(circ)

    legend_patches = [Patch(facecolor=c, edgecolor='black', label=label) for c, label in zip(class_colors, class_labels)]
    ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.15, 1))
    

    
    x_values, y_values = zip(*pos.values())
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)

    x_range = x_max - x_min
    y_range = y_max - y_min
    margin = 0.2 * max(x_range, y_range)  

    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)

    ax.set_aspect('equal')
    ax.axis('off')

    ax.relim()        
    ax.autoscale()     

    if title is not None:
        plt.title(title)

    if results_dir is not None:
        plt.savefig(os.path.join(results_dir, add_title + "soft_classification.png"), bbox_inches='tight', dpi=300)
    
    if show: plt.show()
    plt.close()



def draw_graph_hard_clusters(A, y, class_colors=None, class_labels=None, node_size=None, results_dir=None, show=False, title=None, add_title=""):
    """
    A : (n,n) 
    y : (n,k) 
    class_colors : coleurs, facultatives
    node_size : taille des nodes
    """
    n = len(y)
    k = int(np.max(y))

    if not node_size: 
        node_size = 6000/n

    if not class_colors:
        cmap = plt.get_cmap("tab10")
        class_colors = [mcolors.to_hex(cmap(i % cmap.N)) for i in range(k+1)]
    
    if class_labels is None:
        class_labels = [f"Class {i}" for i in range(k+1)]

    node_colors = [ class_colors[i] for i in y]

    if np.issubdtype(np.array(node_colors).dtype, np.number):
        cmap = plt.get_cmap("tab10")
        node_colors = [cmap(v / max(node_colors)) for v in node_colors]
    

    G = nx.from_numpy_array(A)
    pos = nx.spring_layout(G, seed=0)

    fig, ax = plt.subplots(figsize=(8,8))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4, edge_color='grey')
    nx.draw_networkx_nodes(G, pos,
                           node_color=node_colors,
                           node_size=node_size,
                           edgecolors='black',
                           linewidths=0.5,
                           ax=ax)


    for (x, y) in pos.values():
        circ = plt.Circle((x, y), node_size/30000, fill=False, edgecolor='black', lw=0.5)
        ax.add_patch(circ)

    #legend_patches = [Patch(facecolor=c, edgecolor='black', label=label) for c, label in zip(class_colors, class_labels)]
    #ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    ax.set_aspect('equal')
    ax.axis('off')

    if title is not None:
        plt.title(title)


    if results_dir is not None:
        plt.savefig(os.path.join(results_dir, add_title + "hard_clusters.png"), bbox_inches='tight', dpi=300)
    
    if show: plt.show()
    plt.close()




    


# ---------------------------------------------------------------------
# OBJECTIF (ELBO) 
# ---------------------------------------------------------------------

#Need Pi and eta if neg sampling
def elbo_stub(A, P, mu, logvar, Pi=None, eta=None, 
              use_negative_sampling=False, neg_ratio=5, edges_pos=None, neg_pairs_all=None):
    """
    ELBO avec ou sans negative sampling.
    Si use_negative_sampling=False → version complète O(N^2)
    Si True → edges + neg_samples
    """

    eps = 1e-8 #Prevent from having log(0)

    # =========================================================
    #   CAS 1 : PAS DE NEGATIVE SAMPLING → VERSION COMPLETE
    # =========================================================
    if not use_negative_sampling:
        N = A.shape[0]
        ij = np.triu_indices(N, 1)
        ll = (A[ij] * torch.log(P[ij] + eps) 
             + (1-A[ij]) * torch.log(1-P[ij] + eps)).sum()

    # =========================================================
    #   CAS 2 : NEGATIVE SAMPLING (utilise Pi et eta)
    # =========================================================
    else:
        # ---------- 1) positive edges ----------
        assert edges_pos is not None and neg_pairs_all is not None
        eps = 1e-8

        # ---------- 1) positive edges (déjà pré-calculées) ----------
        i_pos = edges_pos[:, 0]
        j_pos = edges_pos[:, 1]

        eta_i = eta[i_pos]          # (E_pos, Q)
        eta_j = eta[j_pos]          # (E_pos, Q)

        pij_pos = torch.sum(eta_i * (eta_j @ Pi.T), dim=1).clamp(eps, 1 - eps)
        pos_ll = torch.log(pij_pos).sum()

        # ---------- 2) negative edges (sous-échantillonnées) ----------
        num_pos = edges_pos.size(0)
        num_neg = min(neg_ratio * num_pos, neg_pairs_all.size(0))

        idx = torch.randperm(neg_pairs_all.size(0), device=neg_pairs_all.device)[:num_neg]
        neg_edges = neg_pairs_all[idx]   # (num_neg, 2)

        i_neg = neg_edges[:, 0]
        j_neg = neg_edges[:, 1]

        eta_i_neg = eta[i_neg]
        eta_j_neg = eta[j_neg]

        pij_neg = torch.sum(eta_i_neg * (eta_j_neg @ Pi.T), dim=1).clamp(eps, 1 - eps)
        neg_ll = torch.log(1 - pij_neg).sum()

        ll = pos_ll + neg_ll


    # Variance isotrope → logvar shape (N,1)
    sigma2 = torch.exp(logvar)   # (N,1)
    kl = 0.5 * (sigma2 + mu**2 - 1 - logvar).sum()

    return ll - kl

def elbo_stub_old(A, P, mu, logvar):
    ij = np.triu_indices(A.shape[0], 1)
    ll = (A[ij] * torch.log(P[ij]) + (1-A[ij]) * torch.log(1-P[ij])).sum()

    # Variance isotrope → logvar shape (N,1)
    sigma2 = torch.exp(logvar)   # (N,1)
    kl = 0.5 * (sigma2 + mu**2 - 1 - logvar).sum()

    return ll - kl


# ---------------------------------------------------------------------
# ENTRAÎNEMENT (Algorithme 1)
# ---------------------------------------------------------------------

def train_deep_lpbm_GCN(A, Q, config, seed=None, results_dir=None, negetive_sampling=False, neg_ratio=5, hidden_override=None, lr_override=None):
    if seed is None: 
        seed = config['RANDOM_STATE']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A_t = torch.tensor(A, dtype=torch.float32, device=device)
    best = {"elbo": -np.inf, "eta": None, "Pi": None}
    if negetive_sampling:
        # 1) arêtes positives
        edges_pos = torch.nonzero(A_t.triu(diagonal=1), as_tuple=False)   # (E_pos, 2)

        N = A_t.size(0)
        # 2) toutes les non-arêtes possibles
        iu, ju = torch.triu_indices(N, N, offset=1, device=device)
        mask = (A_t[iu, ju] == 0)
        neg_pairs_all = torch.stack([iu[mask], ju[mask]], dim=1)          # (E_neg, 2)
    else:
        edges_pos = None
        neg_pairs_all = None

    for s in range(config['NUM_SEEDS']):
        params = {"Q": Q, "seed": seed + s, "device": device,
                  "hidden": hidden_override if hidden_override is not None else config['HIDDEN_LAYERS_GCN'], "init_lr": lr_override if lr_override is not None else config['LEARNING_RATE']}
        
        init_encoder_phase(A, Q, params, config, results_dir)

        gcn = params["gcn"]
        Pi_tilde = params["Pi_tilde0"]
        #Pi_tilde = torch.zeros((Q, Q), dtype=torch.float32, device=device, requires_grad=True)
        opt = torch.optim.Adam(list(gcn.parameters()) + [Pi_tilde], lr=config['LEARNING_RATE'])

        elbo_history = []
        gcn.train()
        for _ in range(config['MAX_EPOCHS'] ):
            opt.zero_grad()
            X = torch.eye(A_t.size(0), dtype=torch.float32, device=device)
            mu, logvar = gcn(A_t, X)
            z = reparameterize(mu, logvar)
            eta = softmax_logits_to_eta(z) if negetive_sampling else None #Nécessaire que pour le negative sampling
            Pi = unconstrained_Pi_to_Pi(Pi_tilde)
            P = None if negetive_sampling else decoder_prob(z, Pi) #On évite un calcul en O(N*N) avec le negative sampling
            elbo = elbo_stub(
                A_t, P, mu, logvar,
                Pi=Pi,
                eta=eta,
                use_negative_sampling=negetive_sampling,
                neg_ratio=neg_ratio,
                edges_pos=edges_pos,
                neg_pairs_all=neg_pairs_all
            )

            loss = -elbo
            loss.backward()
            opt.step()
            elbo_history.append(float(elbo.detach().cpu()))

        # Évaluation finale
        gcn.eval()
        with torch.no_grad():
            X = torch.eye(A_t.size(0), dtype=torch.float32, device=device)
            mu, logvar = gcn(A_t, X)
            z = reparameterize(mu, logvar)
            eta = softmax_logits_to_eta(z).cpu().numpy()
            Pi = unconstrained_Pi_to_Pi(Pi_tilde).cpu().numpy()

        last_elbo = elbo_history[-1]
        if last_elbo > best["elbo"]:
            best.update({"elbo": last_elbo, "eta": eta, "Pi": Pi, "history": elbo_history})

    # Sauvegarde des figures (si demandé)
    if results_dir is not None:
        save_all_figures(results_dir, best, Q)

    return best



def train_deep_lpbm_GCNEncoder(A, Q, config, seed=None, results_dir=None, negetive_sampling=False, neg_ratio=5, hidden_override=None, lr_override=None):
    """
    Entraîne un modèle Deep LPBM avec encodeur GCN probabiliste (version PyG).
    - Utilise la nouvelle classe GCN basée sur GCNConv (entrée sparse edge_index).
    """

    if seed is None:
        seed = config['RANDOM_STATE']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    
    # Convertit la matrice dense A en format edge_index pour PyG
    A_t = torch.tensor(A, dtype=torch.float32, device=device)
    N = A_t.size(0)
    edge_index, _ = dense_to_sparse(A_t)     # (2, E)
    if negetive_sampling:
        # 1) arêtes positives
        edges_pos = torch.nonzero(A_t.triu(diagonal=1), as_tuple=False)   # (E_pos, 2)

        # 2) toutes les non-arêtes possibles
        iu, ju = torch.triu_indices(N, N, offset=1, device=device)
        mask = (A_t[iu, ju] == 0)
        neg_pairs_all = torch.stack([iu[mask], ju[mask]], dim=1)          # (E_neg, 2)
    else:
        edges_pos = None
        neg_pairs_all = None
    # Features : vecteur identité ou constantes (selon ton choix)
    X = torch.eye(N, dtype=torch.float32, device=device)

    best = {"elbo": -np.inf, "eta": None, "Pi": None}
    elbo_time = 0
    gcn_time = 0
    backward_time = 0
    eval_time = 0
    for s in range(config['NUM_SEEDS']):
        torch.manual_seed(seed + s)

        # Initialisation du modèle GCN (défini ailleurs avec GCNConv)
        params = {"Q": Q, "seed": seed + s, "device": device,
                  "hidden": hidden_override[0] if hidden_override is not None else config['HIDDEN_LAYERS_GCN'][0], "init_lr": lr_override if lr_override is not None else config['LEARNING_RATE']}
        init_encoder_phase_GCNEncoder(A, Q, params, config, results_dir)
        gcn = params["gcn"]  # ta classe GCN (avec GCNConv)

        # Paramètres du LPBM
        Pi_tilde = params["Pi_tilde0"]
        #Pi_tilde = torch.zeros((Q, Q), dtype=torch.float32, device=device, requires_grad=True)
        opt = torch.optim.Adam(list(gcn.parameters()) + [Pi_tilde], lr=config['LEARNING_RATE'])

        elbo_history = []

        # === Phase d'entraînement ===
        gcn.train()
        for _ in range(config['MAX_EPOCHS'] ):
            opt.zero_grad()

            # Passage GCN → µ et logσ²
            t0 = time.time()
            mu, logvar = gcn(X, edge_index)
            t1 = time.time()
            # Rééchantillonnage et calcul ELBO
            z = reparameterize(mu, logvar)
            eta = softmax_logits_to_eta(z) if negetive_sampling else None #Nécessaire que pour le negative sampling
            Pi = unconstrained_Pi_to_Pi(Pi_tilde)
            P = None if negetive_sampling else decoder_prob(z, Pi) #On évite un calcul en O(N*N) avec le negative sampling
            t2 = time.time()
            elbo = elbo_stub(
                A_t, P, mu, logvar,
                Pi=Pi,
                eta=eta,
                use_negative_sampling=negetive_sampling,
                neg_ratio=neg_ratio,
                edges_pos=edges_pos,
                neg_pairs_all=neg_pairs_all
            )
            t3 = time.time()
            # Backpropagation
            loss = -elbo
            loss.backward()
            opt.step()
            t4 = time.time()
            gcn_time += t1-t0
            elbo_time += t3-t2
            backward_time += t4-t3
            elbo_history.append(float(elbo.detach().cpu()))

        t5 = time.time()
        # === Phase d'évaluation ===
        gcn.eval()
        with torch.no_grad():
            mu, logvar = gcn(X, edge_index)
            z = reparameterize(mu, logvar)
            eta = softmax_logits_to_eta(z).cpu().numpy()
            Pi = unconstrained_Pi_to_Pi(Pi_tilde).cpu().numpy()

        last_elbo = elbo_history[-1]
        if last_elbo > best["elbo"]:
            best.update({
                "elbo": last_elbo,
                "eta": eta,
                "Pi": Pi,
                "history": elbo_history
            })
        t6 = time.time()
        eval_time += t6-t5

    print(f"GCN: {gcn_time:.4f}  ELBO: {elbo_time:.4f}  backward: {backward_time:.4f} eval: {eval_time:.4f}")
    # === Sauvegarde des figures ===
    if results_dir is not None:
        save_all_figures(results_dir, best, Q)
    print(next(params["gcn"].parameters()).device)
    print(f"[Deep LPBM + GCNConv] Q={Q} — ELBO finale : {best['elbo']:.4f}")
    return best

# ---------------------------------------------------------------------
# SÉLECTION DE MODÈLE (Q) — critères info
# ---------------------------------------------------------------------
def loglikelihood_A_given_eta_Pi(A, eta, Pi):
    """
    ln p(A | η, Π) (Bernoulli indépendante par arête, non orienté).
    Rappel article : Sec. 4.3 (critères info).
    """
    eps = 1e-12
    # p_ij = η_i^T Π η_j
    P = eta @ Pi @ eta.T
    P = np.clip(P, eps, 1 - eps)  # stabilité num. pour log
    # somme sur i<j (graphe non orienté, pas de boucles)
    ij = np.triu_indices_from(A, k=1)
    ll = (A[ij] * np.log(P[ij]) + (1 - A[ij]) * np.log(1 - P[ij])).sum()
    return float(ll)



def compute_AIC_BIC_ICL(A, eta, Pi):
    """
    AIC, BIC, ICL (avec η fixé, comme dans l’article).
    Rappel article : Sec. 4.3 (+ résultats Sec. 5.2).
    """
    N = A.shape[0]
    Q = eta.shape[1]
    ll_A = loglikelihood_A_given_eta_Pi(A, eta, Pi)
    nu_NQ = Q*(Q+1) + N*(Q-1)
    nu_NQ_Pi = 0.5*Q*(Q+1)
    Nobs = 0.5*N*(N-1)

    AIC = ll_A - nu_NQ
    BIC = ll_A - 0.5*nu_NQ*np.log(Nobs)
    # ICL approx (remplacer par ln p(A,Z|θ) si tu utilises une partition dure)
    ICL = ll_A - 0.5*nu_NQ_Pi*np.log(Nobs)
    return AIC, BIC, ICL

def collapse_small_clusters(eta, min_size_ratio=0.03):
    """
    Prend une matrice d'appartenance partielle η (N×Q)
    → détecte les clusters trop petits (<3%)
    → redistribue leurs nœuds vers le cluster suivant le plus probable
    → retourne une nouvelle η_collapsed (même taille N×Q)
    """
    N, Q = eta.shape
    min_size = max(1, int(min_size_ratio * N))

    # assignation dure
    y = eta.argmax(axis=1)
    
    # taille des clusters
    counts = np.bincount(y, minlength=Q)
    small = np.where(counts < min_size)[0]

    if len(small) == 0:
        return eta  # rien à faire

    eta_new = eta.copy()

    for sc in small:
        nodes = np.where(y == sc)[0]
        if len(nodes) == 0:
            continue

        # pour ces noeuds, prendre la 2e proba la plus forte
        for i in nodes:
            # trier par probas décroissantes
            order = np.argsort(eta[i])[::-1]
            # prendre la plus grande ≠ sc
            for alt in order:
                if alt != sc:
                    eta_new[i, sc] = 0.0
                    eta_new[i, alt] = 1.0
                    break

    # renormalisation simple (pas toujours nécessaire car one-hot)
    eta_new = eta_new / eta_new.sum(axis=1, keepdims=True)
    print("collapsed!")
    return eta_new


def model_selection_over_Q(A, Q_list, config, subject_name="subject", seed=None, CLASS_GCN = None, negetive_sampling=False, neg_ratio=5, hidden_override=HIDDEN_LAYERS_GCN,
    lr_override=LEARNING_RATE):
    """
    Essaie plusieurs valeurs de Q et renvoie le meilleur modèle selon l'AIC.
    Crée un sous-dossier results/<subject_name>/Q_<Q>/ pour chaque entraînement.
    """
    if seed is None:
        seed = config['RANDOM_STATE']

    if CLASS_GCN is None:
        CLASS_GCN = config['CLASS_GCN'] 

    results_dir_base = os.path.join("results", subject_name)
    os.makedirs(results_dir_base, exist_ok=True)

    results = []
    AIC_log = []
    for Q in Q_list:
        print(f"\n=== Entraînement pour Q = {Q} ===")
        results_dir_Q = os.path.join(results_dir_base, f"Q_{Q}")
        if config['CLASS_GCN']  == "GCN":
            fit = train_deep_lpbm_GCN(A, Q, config, seed=seed, results_dir=results_dir_Q, negetive_sampling=negetive_sampling, neg_ratio=neg_ratio, hidden_override=hidden_override, lr_override=lr_override)
        if config['CLASS_GCN']  == "GCNEncoder":
            fit = train_deep_lpbm_GCNEncoder(A, Q, config, seed=seed, results_dir=results_dir_Q, negetive_sampling=negetive_sampling, neg_ratio=neg_ratio, hidden_override=hidden_override, lr_override=lr_override)
        
        # collapse éventuel des petits clusters
        eta_collapsed = collapse_small_clusters(fit["eta"], min_size_ratio=0.03)
        fit["eta"] = eta_collapsed

        # recalcul des critères d'information
        AIC, BIC, ICL = compute_AIC_BIC_ICL(A, fit["eta"], fit["Pi"])
        AIC_log.append(AIC)
        fit.update({"Q": Q, "AIC": AIC, "BIC": BIC, "ICL": ICL})
        results.append(fit)

        #draw_graph_with_probabilities(A, fit["eta"], results_dir=results_dir_Q)

        y_hat = fit["eta"].argmax(axis=1)
        #draw_graph_hard_clusters(A, y_hat, results_dir=results_dir_Q)


        # Sauvegarde rapide des scores numériques
        with open(os.path.join(results_dir_Q, "scores.txt"), "w") as f:
            f.write(f"AIC: {AIC:.3f}\nBIC: {BIC:.3f}\nICL: {ICL:.3f}\nELBO: {fit['elbo']:.3f}\n")

    # Sélection du meilleur modèle selon AIC (comme recommandé dans l’article)
    best = max(results, key=lambda d: d[config['METRIC']])
    best_Q = best["Q"]

    plt.figure(figsize=(6, 4))

    plt.plot(Q_list, AIC_log, marker='o')
    plt.vlines(best_Q, ymin=min(AIC_log), ymax=max(AIC_log),
            color='red', linestyle='--', label='Best Q')

    plt.title("AIC values for each tested Q")
    plt.xlabel("Number of clusters Q")
    plt.ylabel("AIC")
    plt.legend()
    plt.tight_layout()

    os.makedirs(results_dir_base, exist_ok=True)
    plt.savefig(os.path.join(results_dir_base, "AIC_over_Q.png"))
    plt.close()


    return best, results


# ---------------------------------------------------------------------
# ÉVALUATION
# ---------------------------------------------------------------------
def hard_clusters_from_eta(eta):
    """
    Partition dure par argmax(η) (utile pour ARI).
    Rappel article : métriques (Sec. 5 / App. C.1).
    """
    return eta.argmax(axis=1)

def H_partial_memberships_score(eta_true, eta_hat):
    """
    Métrique H (Sec. 5) : distance L1 normalisée entre co-appartenances U*=η*η*^T et Û=η̂η̂^T,
    sommée sur i<j (paires uniques), invariante aux permutations de clusters.
    """
    eps = 1e-12
    
    U_true = eta_true @ eta_true.T
    U_hat  = eta_hat  @ eta_hat.T
    N = eta_true.shape[0]
    tri = np.triu_indices(N, k=0)  # i<j, exclut la diagonale
    diff = np.abs(U_true - U_hat)[tri]
    return np.sqrt(2.0/(N*(N-1)) * diff.sum()   )



def cluster_counts(labels):
    """Renvoie un dictionnaire {cluster_id: count}."""
    uniq, counts = np.unique(labels, return_counts=True)
    return dict(zip(uniq, counts))


def best_label_permutation(y_true, y_pred):
    """
    Trouve la meilleure permutation des labels prédits pour matcher y_true.
    """
    classes_true = np.unique(y_true)
    classes_pred = np.unique(y_pred)
    n_true, n_pred = len(classes_true), len(classes_pred)
    cost_matrix = np.zeros((n_true, n_pred))
    for i, c_true in enumerate(classes_true):
        for j, c_pred in enumerate(classes_pred):
            # coût = nombre d’erreurs (on veut le minimiser)
            cost_matrix[i, j] = np.sum((y_true == c_true) & (y_pred != c_pred))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {classes_pred[j]: classes_true[i] for i, j in zip(row_ind, col_ind)}
    return mapping

def relabel_predictions(y_pred, mapping):
    """Applique la correspondance optimale (mapping) aux labels prédits."""
    y_aligned = np.copy(y_pred)
    for old, new in mapping.items():
        y_aligned[y_pred == old] = new
    return y_aligned



def confusion_matrix_permuted(y_true, y_pred):
    mapping = best_label_permutation(y_true, y_pred)
    y_aligned = relabel_predictions(y_pred, mapping)
    cm = confusion_matrix(y_true, y_aligned)
    return cm, mapping, y_aligned



def show_confusion_matrix(y_true, y_pred):
    cm, mapping = confusion_matrix_permuted(y_true, y_pred)
    df = pd.DataFrame(
        cm,
        index=[f"True {c}" for c in np.unique(y_true)],
        columns=[f"Pred {c}" for c in np.unique(y_true)]
    )
    print("🔹 Correspondance optimale :", mapping)
    print("\n=== Matrice de confusion (après permutation) ===")
    print(df)

def global_clustering_scores(y_true, y_pred):
    """
    Calcule ARI et NMI (robustes aux permutations de labels).
    """
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    print(f"ARI : {ari:.3f}")
    print(f"NMI : {nmi:.3f}")
    return ari, nmi


def evaluate_clustering(y_true=None, y_pred=None,
                        eta_true=None, eta_hat=None,
                        Pi_true=None, Pi_pred=None):
    """
    Évalue un clustering dur ou partiel + compare Pi_true / Pi_pred si fournis.
    """
    print("=== Évaluation du modèle ===")

    # ---- Comparaison des matrices Π si présentes ----
    if (Pi_true is not None) and (Pi_true.shape == Pi_pred.shape) :
        diff = np.abs(Pi_true - Pi_pred).mean()
        print(f"\n=== Comparaison Π ===")
        print(f"Différence moyenne |Π_true - Π_pred| = {diff:.4f}")
        print(f"Shape Π_true = {Pi_true.shape} | Shape Π_pred = {Pi_pred.shape}")

    # ---- Cas des appartenances partielles ----
    if eta_true is not None and eta_hat is not None:
        H = H_partial_memberships_score(eta_true, eta_hat)
        y_true_hard = hard_clusters_from_eta(eta_true)
        y_pred_hard = hard_clusters_from_eta(eta_hat)
        print(f"\nScore H (appartenances partielles) = {H:.4f}")
        y_true, y_pred = y_true_hard, y_pred_hard

    # ---- Cas des labels durs ----
    if y_true is not None and y_pred is not None:
        cm, mapping, y_aligned = confusion_matrix_permuted(y_true, y_pred)

        labels = sorted(set(y_true) | set(y_pred))

        df = pd.DataFrame(
            cm,
            index=[f"True {c}" for c in labels],
            columns=[f"Pred {c}" for c in labels]
        )


        ari = adjusted_rand_score(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)

        print("\n=== Matrice de confusion ===")
        print(df)

        print("\n=== Scores globaux ===")
        print(f"ARI = {ari:.3f}")
        print(f"NMI = {nmi:.3f}")

        for c in np.unique(y_true):
            mask = y_true == c
            correct = np.sum(y_aligned[mask] == c)
            total = np.sum(mask)
            print(f"Cluster {c}: {correct}/{total} corrects ({100*correct/total:.1f}%)")

    elif eta_true is None or eta_hat is None:
        print(" Fournis au moins y_true/y_pred ou eta_true/eta_hat.")


def save_node_assignments(y_true, y_pred, mapping, subject_name,  results_root="results"):
    """
    Enregistre les assignations vraies/prédites des noeuds dans le dossier results/<patient>/assignments/.
    Le nom du fichier correspond à la matrice observée (ex: results/subject01/assignments/subject01.json).
    """
    # --- Préparation du dossier ---
    out_dir = os.path.join(results_root, subject_name, "assignments")
    os.makedirs(out_dir, exist_ok=True)

    # --- Applique la permutation optimale aux prédictions ---
    y_aligned = relabel_predictions(y_pred, mapping)

    # --- Construit un tableau clair ---
    data = {
        "node_id": np.arange(len(y_true)).tolist(),
        "true_label": y_true.astype(int).tolist(),
        "pred_label_raw": y_pred.astype(int).tolist(),
        "pred_label_aligned": y_aligned.astype(int).tolist()
    }
    df = pd.DataFrame(data)

    # --- Noms des fichiers ---
    json_path = os.path.join(out_dir, f"{subject_name}.json")
    csv_path  = os.path.join(out_dir, f"{subject_name}.csv")

    # --- Sauvegarde ---
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    df.to_csv(csv_path, index=False)

    print(f"\nAssignations enregistrées dans :\n  {json_path}\n  {csv_path}")

# ---------------------------------------------------------------------
# PIPELINE EXEMPLE (consomme tes .npy)
# ---------------------------------------------------------------------

def save_Pi_comparison(Pi_true, Pi_pred, outpath):
    """
    Crée et sauvegarde une figure comparant Π_true et Π_pred.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axes[0].imshow(Pi_true, cmap="viridis")
    axes[0].set_title("Π vraie (ground truth)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(Pi_pred, cmap="viridis")
    axes[1].set_title("Π prédite (Deep LPBM)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    plt.suptitle("Comparaison des matrices Π", fontsize=14)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main( config={} ):

    config_zero = {
        'MODE' : "assortative", 
        'DATA_DIR' : "data_synthetic/assortative" ,
        'RANDOM_STATE' : 42,
        'HIDDEN_LAYERS_GCN' : [ 32 ], 
        'SPLITLAYER' : False,
        'MAX_EPOCHS_INIT' : 300,     # Phase init encodeur (Algorithme 1)
        'MAX_EPOCHS' : 600,          # Phase estimation (Algorithme 1)
        'LEARNING_RATE' : 1e-2 ,   # Adam lr=0.01 dans l'article
        'NUM_SEEDS' : 5,           # on garde le meilleur ELBO
        'CLASS_GCN' : "GCNEncoder",  #  "GCNEncoder" , 
        'SUBJECT_IDX' : 0,
        'METRIC' : "AIC",
        'Q_list' : [3, 4, 5, 6, 7]       ,
        'NEG_RATIO': 5,
        'USE_NEGATIVE_SAMPLING' : False  }


    for key in config_zero.keys():
        if key not in config:
            config[key] = config_zero[key]


    # --- 2. Sélection du sujet ---
    
    A_filename = f"A_{config['SUBJECT_IDX']:03d}.npy"     
    A_path = os.path.join(config['DATA_DIR'], A_filename) 

    A = np.load(A_path)
    
    subject_name = os.path.splitext(os.path.basename(A_path))[0].replace("A_", "")
    print(f"Sujet sélectionné : {subject_name} dans le dossier {config['DATA_DIR']}")

    # --- 3. Sélection du nombre de clusters ---
    Q_list = config['Q_list']
    best, all_results = model_selection_over_Q(A, Q_list, config, subject_name=config['MODE'] + subject_name, negative_sampling=config['USE_NEGATIVE_SAMPLING'], neg_ratio=config['NEG_RATIO'])

    # --- 4. Résumé du meilleur modèle ---
    print("\n=== Meilleur modèle ===")
    print(f"Q optimal (AIC) : {best['Q']}")
    print(f"AIC : {best['AIC']:.2f}   |   BIC : {best['BIC']:.2f}   |   ICL : {best['ICL']:.2f}")
    print(f"η shape : {best['eta'].shape}   |   Π shape : {best['Pi'].shape}")


    # --- 5. Partition dure (clusters prédits) ---
    y_hat = hard_clusters_from_eta(best["eta"])
    Pi = best['Pi']

    # --- 6. Recherche d’un fichier y_* pour vérifier s’il s’agit de données synthétiques ---
    y_path = os.path.join(config['DATA_DIR'], os.path.basename(A_path).replace("A_", "y_"))
    Pi_path = os.path.join(config['DATA_DIR'], os.path.basename(A_path).replace("A_", "Pi_"))
    eta_path = os.path.join(config['DATA_DIR'], os.path.basename(A_path).replace("A_", "eta_"))
    
    if os.path.exists(y_path):
        print("\n=== Données synthétiques détectées ===")
        y = np.load(y_path).astype(int).ravel()
        K_true = np.unique(y).size
        print(f"K vrai : {K_true}   |   Q trouvé : {best['Q']}")

        # --- 7. Création de η* (one-hot de y vrai) ---
        eta_true = np.load(eta_path)
        Pi_true = np.load(Pi_path)

        # --- 8. Évaluation ---
        evaluate_clustering(
            y_true=y,
            y_pred=y_hat,
            eta_true=eta_true,
            eta_hat=best["eta"],
            Pi_true=Pi_true,
            Pi_pred=Pi
        )

        # --- 9. Sauvegarde des assignations ---
        save_node_assignments(
            y_true=y,
            y_pred=y_hat,
            mapping=best_label_permutation(y, y_hat),
            subject_name=config['MODE'] + subject_name,
            results_root="results"
        )

        # --- 10. Sauvegarde de Π vraie et prédite ---
    
        results_dir = os.path.join("results", config['MODE'] + subject_name)
        os.makedirs(results_dir, exist_ok=True)

        fig_path = os.path.join(results_dir, "Pi_comparison.png")
        save_Pi_comparison(Pi_true, Pi, fig_path)
    else:
        # --- Cas réel : pas de vérité terrain ---
        print("\n=== Données réelles ===")
        print("Aucune vérité terrain disponible.")
        print("Comptages par cluster prédits :", cluster_counts(y_hat))
    
    return {"z": y_hat, "K": best['Q'], "eta": best['eta']}

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Temps d'exécution : {end - start:.3f} secondes")
