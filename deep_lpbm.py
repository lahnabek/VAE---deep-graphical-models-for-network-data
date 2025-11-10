# deep_lpbm_minimal.py
# Squelette minimal pour répliquer le modèle à partir de matrices .npy déjà prêtes.
# Pas de classes, scikit-learn quand utile, et TODO à compléter.

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

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Patch
import matplotlib.colors as mcolors




from class_GCN import *
from class_GCNEncoder import *
# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

DATA_DIR =  "miscdata" #"data_numpy_synthetic" 
RANDOM_STATE = 42
HIDDEN_LAYERS_GCN = 32
MAX_EPOCHS_INIT = 300     # Phase init encodeur (Algorithme 1)
MAX_EPOCHS = 400          # Phase estimation (Algorithme 1)
LEARNING_RATE = 1e-2      # Adam lr=0.01 dans l'article
NUM_SEEDS = 10            # on garde le meilleur ELBO
CLASS_GCN = "GCNEncoder"  # ou "GCN"
# ---------------------------------------------------------------------
# PRÉTRAITEMENT (init k-means etc.)
# ---------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def kmeans_initial_memberships(A, Q, results_dir=None):
    """
    - A : NxN binary symmetric adjacency matrix
    - Q : number of clusters
    Initialisation douce des appartenances par K-Means sur A.
    Sauvegarde dans results_dir :
        - kmeans_init.csv : labels et one-hot
        - kmeans_init_hist.png : histogramme des clusters initiaux
    """
    km = KMeans(n_clusters=Q, n_init="auto", random_state=RANDOM_STATE)
    labels = km.fit_predict(A)
    N = A.shape[0]

    # One-hot encoding (matrice d’appartenance initiale)
    C = np.zeros((N, Q))
    C[np.arange(N), labels] = 1.0

    # --- Sauvegardes ---
    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)

        # 1) Sauvegarde CSV
        df = pd.DataFrame({
            "node_id": np.arange(N),
            "kmeans_label": labels.astype(int)
        })
        csv_path = os.path.join(results_dir, "kmeans_init.csv")
        df.to_csv(csv_path, index=False)

        # 2) Histogramme des clusters
        plt.figure()
        plt.hist(labels, bins=np.arange(Q + 1) - 0.5, rwidth=0.8)
        plt.xlabel("Cluster K-Means initial")
        plt.ylabel("Nombre de nœuds")
        plt.title(f"Répartition initiale K-Means (Q={Q})")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "kmeans_init_hist.png"))
        plt.close()

    return C

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


# ---------------------------------------------------------------------
# UTILS
# ---------------------------------------------------------------------

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

def init_encoder_phase(A, Q, params, results_dir=None):
    """
    Phase d'initialisation de l’encodeur GCN (Algorithme 1 / App. B.2).
    - µ ≈ z₀ (issu de KMeans)
    - logσ² ≈ log(0.01)

    Sauvegarde (dans results_dir) :
        - répartition initiale KMeans (CSV + histogramme)
        - courbe de perte d'initialisation init_loss.png
    """
    device   = torch.device(params.get("device", "cpu"))
    hidden   = int(params.get("hidden", HIDDEN_LAYERS_GCN))
    init_lr  = float(params.get("init_lr", LEARNING_RATE))
    seed     = int(params.get("seed", RANDOM_STATE))
    tau      = float(params.get("init_tau", 1e-3))

    torch.manual_seed(seed)
    N = A.shape[0]

    # --- Création du dossier (si demandé) ---
    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)

    # --- Modèle GCN (créé une fois et stocké dans params)
    gcn = params.get("gcn")
    if gcn is None:
        gcn = GCN(in_feats=1, hidden=hidden, out_mu=Q-1, out_lv=1).to(device)
        params["gcn"] = gcn

    # --- Optimiseur pour la phase d'init
    opt = params.get("opt_init")
    if opt is None:
        opt = torch.optim.Adam(gcn.parameters(), lr=init_lr)
        params["opt_init"] = opt

    # --- Tenseurs d’entrée
    A_t = torch.tensor(A, dtype=torch.float32, device=device)
    X   = torch.ones((N, 1), dtype=torch.float32, device=device)

    # --- Initialisation douce via KMeans ---
    C = kmeans_initial_memberships(A, Q, results_dir=results_dir)
    z0_np   = np.log((C[:, :Q-1] + tau) / (C[:, [Q-1]] + tau))
    z0      = torch.tensor(z0_np, dtype=torch.float32, device=device)
    lv_star = torch.full((N, 1), np.log(0.01), dtype=torch.float32, device=device)

    # --- Boucle d’entraînement (phase d’init) ---
    loss_history = []
    gcn.train()
    for _ in range(MAX_EPOCHS_INIT):
        opt.zero_grad()
        mu, logvar = gcn(A_t, X)
        loss_mu = F.mse_loss(mu, z0)
        loss_lv = F.mse_loss(logvar, lv_star)
        loss = loss_mu + loss_lv
        loss.backward()
        opt.step()
        loss_history.append(float(loss.detach().cpu()))

    last_loss = loss_history[-1] if loss_history else None

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


def init_encoder_phase_GCNEncoder(A, Q, params, results_dir=None):
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
    hidden   = int(params.get("hidden", HIDDEN_LAYERS_GCN))
    init_lr  = float(params.get("init_lr", LEARNING_RATE))
    seed     = int(params.get("seed", RANDOM_STATE))
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
    gcn = params.get("gcn")
    if gcn is None:
        gcn = GCNEncoder(in_feats=N, hidden=hidden, out_mu=Q-1, out_lv=1).to(device)
        params["gcn"] = gcn

    # --- Optimiseur pour la phase d'init
    opt = params.get("opt_init")
    if opt is None:
        opt = torch.optim.Adam(gcn.parameters(), lr=init_lr)
        params["opt_init"] = opt

    # --- Initialisation douce via KMeans ---
    C = kmeans_initial_memberships(A, Q, results_dir=results_dir)
    z0_np   = np.log((C[:, :Q-1] + tau) / (C[:, [Q-1]] + tau))
    z0      = torch.tensor(z0_np, dtype=torch.float32, device=device)
    lv_star = torch.full((N, 1), np.log(0.01), dtype=torch.float32, device=device)

    # --- Boucle d’entraînement (phase d’init) ---
    loss_history = []
    gcn.train()
    for _ in range(MAX_EPOCHS_INIT):
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


# -----------------------------------------------------------
# REPRESENTATION
# -----------------------------------------------------------


def draw_graph_with_probabilities(A, eta, class_colors=None, class_labels=None, node_radius=None, results_dir=None):
    """
    A : (n,n) 
    eta : (n,k) 
    class_colors : coleurs, facultatives
    node_radius : taille des nodes
    """
    n, k = eta.shape

    if not node_radius:
        node_radius = 1.5/n


    if not class_colors:
        cmap = plt.get_cmap("tab10")
        class_colors = [cmap(i % cmap.N) for i in range(k)]

    
    if class_labels is None:
        class_labels = [f"Class {i}" for i in range(k)]


    print('col', class_colors, 'lab', class_labels)

    G = nx.from_numpy_array(A)
    pos = nx.spring_layout(G, seed=0)

    fig, ax = plt.subplots(figsize=(8,8))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4)
    
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

    if results_dir is not None:
        plt.savefig(os.path.join(results_dir, "soft_classification.png"), bbox_inches='tight', dpi=300)
    
    plt.show()
    plt.close()



def draw_graph_hard_clusters(A, y, class_colors=None, class_labels=None, node_size=None, results_dir=None):
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
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4)
    nx.draw_networkx_nodes(G, pos,
                           node_color=node_colors,
                           node_size=node_size,
                           edgecolors='black',
                           linewidths=0.5,
                           ax=ax)


    for (x, y) in pos.values():
        circ = plt.Circle((x, y), node_size/30000, fill=False, edgecolor='black', lw=0.5)
        ax.add_patch(circ)



    legend_patches = [Patch(facecolor=c, edgecolor='black', label=label) for c, label in zip(class_colors, class_labels)]
    ax.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.15, 1))

    
    ax.set_aspect('equal')
    ax.axis('off')


    if results_dir is not None:
        plt.savefig(os.path.join(results_dir, "hard_clusters.png"), bbox_inches='tight', dpi=300)
    
    plt.show()
    plt.close()




    


# ---------------------------------------------------------------------
# OBJECTIF (ELBO) 
# ---------------------------------------------------------------------


def elbo_stub(A, P, mu, logvar):
    """
    - A      : N×N binaire symétrique
    - P      : N×N probas d’arête (sorties du décodeur)
    - mu     : N×(Q−1) (moyennes de q)
    - logvar : N×1 ou N×(Q−1) (log-variances de q)
    Retourne ELBO = log p(A|P) − KL[q(Z)||p(Z)]. Sec. 4.
    """
    # log-vraisemblance Bernoulli sur i<j
    ij = np.triu_indices_from(A, k=1)
    ll = (A[ij] * torch.log(P[ij]) + (1 - A[ij]) * torch.log(1 - P[ij])).sum()

    # KL(q||p) pour q = N(mu, diag(sigma^2)), p = N(0, I)

    sigma2 = torch.exp(logvar).expand_as(mu)
    kl = 0.5 * (sigma2 + mu**2 - 1.0 - logvar.expand_as(mu)).sum()

    return ll - kl # elbo


# ---------------------------------------------------------------------
# ENTRAÎNEMENT (Algorithme 1)
# ---------------------------------------------------------------------

def train_deep_lpbm_GCN(A, Q, seed=RANDOM_STATE, results_dir=None):
    device = torch.device("cpu")
    A_t = torch.tensor(A, dtype=torch.float32, device=device)

    best = {"elbo": -np.inf, "eta": None, "Pi": None}
    for s in range(NUM_SEEDS):
        params = {"Q": Q, "seed": seed + s, "device": device,
                  "hidden": HIDDEN_LAYERS_GCN, "init_lr": LEARNING_RATE}
        
        init_encoder_phase(A, Q, params, results_dir)

        gcn = params["gcn"]

        Pi_tilde = torch.zeros((Q, Q), dtype=torch.float32, device=device, requires_grad=True)
        opt = torch.optim.Adam(list(gcn.parameters()) + [Pi_tilde], lr=LEARNING_RATE)

        elbo_history = []
        gcn.train()
        for _ in range(MAX_EPOCHS):
            opt.zero_grad()
            X = torch.ones((A_t.size(0), 1), dtype=torch.float32, device=device)
            mu, logvar = gcn(A_t, X)
            z = reparameterize(mu, logvar)
            Pi = unconstrained_Pi_to_Pi(Pi_tilde)
            P = decoder_prob(z, Pi)
            elbo = elbo_stub(A_t, P, mu, logvar)

            loss = -elbo
            loss.backward()
            opt.step()
            elbo_history.append(float(elbo.detach().cpu()))

        # Évaluation finale
        gcn.eval()
        with torch.no_grad():
            X = torch.ones((A_t.size(0), 1), dtype=torch.float32, device=device)
            mu, logvar = gcn(A_t, X)
            z = mu
            eta = softmax_logits_to_eta(z).cpu().numpy()
            Pi = unconstrained_Pi_to_Pi(Pi_tilde).cpu().numpy()

        last_elbo = elbo_history[-1]
        if last_elbo > best["elbo"]:
            best.update({"elbo": last_elbo, "eta": eta, "Pi": Pi, "history": elbo_history})

    # Sauvegarde des figures (si demandé)
    if results_dir is not None:
        save_all_figures(results_dir, best, Q)

    return best



def train_deep_lpbm_GCNEncoder(A, Q, seed=RANDOM_STATE, results_dir=None):
    """
    Entraîne un modèle Deep LPBM avec encodeur GCN probabiliste (version PyG).
    - Utilise la nouvelle classe GCN basée sur GCNConv (entrée sparse edge_index).
    """
    device = torch.device("cpu")
    torch.manual_seed(seed)

    # Convertit la matrice dense A en format edge_index pour PyG
    A_t = torch.tensor(A, dtype=torch.float32, device=device)
    edge_index, _ = dense_to_sparse(A_t)     # (2, E)
    N = A_t.size(0)

    # Features : vecteur identité ou constantes (selon ton choix)
    X = torch.eye(N, dtype=torch.float32, device=device)

    best = {"elbo": -np.inf, "eta": None, "Pi": None}

    for s in range(NUM_SEEDS):
        torch.manual_seed(seed + s)

        # Initialisation du modèle GCN (défini ailleurs avec GCNConv)
        params = {"Q": Q, "seed": seed + s, "device": device,
                  "hidden": HIDDEN_LAYERS_GCN, "init_lr": LEARNING_RATE}
        init_encoder_phase_GCNEncoder(A, Q, params, results_dir)
        gcn = params["gcn"]  # ta classe GCN (avec GCNConv)

        # Paramètres du LPBM
        Pi_tilde = torch.zeros((Q, Q), dtype=torch.float32, device=device, requires_grad=True)
        opt = torch.optim.Adam(list(gcn.parameters()) + [Pi_tilde], lr=LEARNING_RATE)

        elbo_history = []

        # === Phase d'entraînement ===
        gcn.train()
        for _ in range(MAX_EPOCHS):
            opt.zero_grad()

            # Passage GCN → µ et logσ²
            mu, logvar = gcn(X, edge_index)

            # Rééchantillonnage et calcul ELBO
            z = reparameterize(mu, logvar)
            Pi = unconstrained_Pi_to_Pi(Pi_tilde)
            P = decoder_prob(z, Pi)
            elbo = elbo_stub(A_t, P, mu, logvar)

            # Backpropagation
            loss = -elbo
            loss.backward()
            opt.step()

            elbo_history.append(float(elbo.detach().cpu()))

        # === Phase d'évaluation ===
        gcn.eval()
        with torch.no_grad():
            mu, logvar = gcn(X, edge_index)
            z = mu
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

    # === Sauvegarde des figures ===
    if results_dir is not None:
        save_all_figures(results_dir, best, Q)

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


def count_parameters(N, Q):
    """
    Comptes ν_{N,Q} et ν_{N,Q,Π} comme dans l’article (Sec. 4.3).
    """
    nu_NQ = Q*(Q+1) + N*(Q-1)
    nu_NQ_Pi = int(0.5*Q*(Q+1))
    Nobs = int(0.5*N*(N-1))
    return nu_NQ, nu_NQ_Pi, Nobs

def compute_AIC_BIC_ICL(A, eta, Pi):
    """
    AIC, BIC, ICL (avec η fixé, comme dans l’article).
    Rappel article : Sec. 4.3 (+ résultats Sec. 5.2).
    """
    N = A.shape[0]
    Q = eta.shape[1]
    ll_A = loglikelihood_A_given_eta_Pi(A, eta, Pi)
    nu_NQ, nu_NQ_Pi, Nobs = count_parameters(N, Q)

    AIC = ll_A - nu_NQ
    BIC = ll_A - 0.5*nu_NQ*np.log(Nobs)
    # ICL approx (remplacer par ln p(A,Z|θ) si tu utilises une partition dure)
    ICL = ll_A - 0.5*nu_NQ_Pi*np.log(Nobs)
    return AIC, BIC, ICL



def model_selection_over_Q(A, Q_list, subject_name="subject", seed=RANDOM_STATE, CLASS_GCN = CLASS_GCN):
    """
    Essaie plusieurs valeurs de Q et renvoie le meilleur modèle selon l'AIC.
    Crée un sous-dossier results/<subject_name>/Q_<Q>/ pour chaque entraînement.
    """
    results_dir_base = os.path.join("results", subject_name)
    os.makedirs(results_dir_base, exist_ok=True)

    results = []

    for Q in Q_list:
        print(f"\n=== Entraînement pour Q = {Q} ===")
        results_dir_Q = os.path.join(results_dir_base, f"Q_{Q}")
        if CLASS_GCN == "GCN":
            fit = train_deep_lpbm_GCN(A, Q, seed=seed, results_dir=results_dir_Q)
        if CLASS_GCN == "GCNEncoder":
            fit = train_deep_lpbm_GCNEncoder(A, Q, seed=seed, results_dir=results_dir_Q)
        
        AIC, BIC, ICL = compute_AIC_BIC_ICL(A, fit["eta"], fit["Pi"])
        fit.update({"Q": Q, "AIC": AIC, "BIC": BIC, "ICL": ICL})
        results.append(fit)

        draw_graph_with_probabilities(A, fit["eta"], results_dir=results_dir_Q)

        y_hat = fit["eta"].argmax(axis=1)
        draw_graph_hard_clusters(A, y_hat, results_dir=results_dir_Q)


        # Sauvegarde rapide des scores numériques
        with open(os.path.join(results_dir_Q, "scores.txt"), "w") as f:
            f.write(f"AIC: {AIC:.3f}\nBIC: {BIC:.3f}\nICL: {ICL:.3f}\nELBO: {fit['elbo']:.3f}\n")

    # Sélection du meilleur modèle selon AIC (comme recommandé dans l’article)
    best = max(results, key=lambda d: d["AIC"])
    best_Q = best["Q"]


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
    # (optionnel) sécure: renormalise au cas où les lignes ne somment pas exactement à 1
    eta_true = eta_true / (eta_true.sum(axis=1, keepdims=True) + eps)
    eta_hat  = eta_hat  / (eta_hat.sum(axis=1, keepdims=True)  + eps)

    U_true = eta_true @ eta_true.T
    U_hat  = eta_hat  @ eta_hat.T
    N = eta_true.shape[0]
    tri = np.triu_indices(N, k=1)  # i<j, exclut la diagonale
    diff = np.abs(U_true - U_hat)[tri]
    return np.sqrt(2.0/(N*(N-1))) * diff.sum()



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


def evaluate_clustering(y_true=None, y_pred=None, eta_true=None, eta_hat=None):
    """
    Évalue un clustering (dure ou partiel).
    Si y_true/y_pred fournis → évalue via ARI, NMI, confusion matrix.
    Si eta_true/eta_hat fournis → évalue via H et ARI/NMI sur les argmax.
    """
    print("=== Évaluation du modèle ===")

    # ---- Cas des appartenances partielles ----
    if eta_true is not None and eta_hat is not None:
        H = H_partial_memberships_score(eta_true, eta_hat)
        y_true_hard = hard_clusters_from_eta(eta_true)
        y_pred_hard = hard_clusters_from_eta(eta_hat)
        print(f" Score H (appartenances partielles) = {H:.4f}")
        y_true, y_pred = y_true_hard, y_pred_hard  # pour continuer les métriques globales

    # ---- Cas des labels durs ----
    if y_true is not None and y_pred is not None:
        cm, mapping, y_aligned = confusion_matrix_permuted(y_true, y_pred)
        df = pd.DataFrame(
            cm,
            index=[f"True {c}" for c in np.unique(y_true)],
            columns=[f"Pred {c}" for c in np.unique(y_true)]
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


def save_node_assignments(y_true, y_pred, mapping, A_path, results_root="results"):
    """
    Enregistre les assignations vraies/prédites des noeuds dans le dossier results/<patient>/assignments/.
    Le nom du fichier correspond à la matrice observée (ex: results/subject01/assignments/subject01.json).
    """
    # --- Préparation du dossier ---
    base = os.path.splitext(os.path.basename(A_path))[0]
    patient_name = base.replace("A_", "")  # ex: A_subject01.npy -> subject01
    out_dir = os.path.join(results_root, patient_name, "assignments")
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
    json_path = os.path.join(out_dir, f"{patient_name}.json")
    csv_path  = os.path.join(out_dir, f"{patient_name}.csv")

    # --- Sauvegarde ---
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    df.to_csv(csv_path, index=False)

    print(f"\nAssignations enregistrées dans :\n  {json_path}\n  {csv_path}")

# ---------------------------------------------------------------------
# PIPELINE EXEMPLE (consomme tes .npy)
# ---------------------------------------------------------------------




def list_npy_graphs(data_dir: str):
    """
    Retourne les chemins des fichiers .npy dont le nom commence par 'A'.
    """
    return sorted(glob.glob(os.path.join(data_dir, "A*.npy")))

def load_A(path: str, as_bool: bool = True, validate: bool = True):
    """
    Charge une matrice A (carrée, symétrique, diag=0 si validate).
    as_bool: binarise A (uint8) si True.
    """
    A = np.load(path)
    if validate:
        assert A.ndim == 2 and A.shape[0] == A.shape[1], "A doit être carrée."
        if np.issubdtype(A.dtype, np.floating):
            assert np.allclose(A, A.T, atol=1e-8), "A doit être symétrique."
        else:
            assert (A == A.T).all(), "A doit être symétrique."
        assert np.allclose(np.diag(A), 0), "diag(A) doit être nulle."
    if as_bool:
        A = (A != 0).astype(np.uint8)
    return A

def main():
    # --- 1. Chargement des fichiers ---
    files = list_npy_graphs(DATA_DIR)
    assert len(files) > 0, f"Aucun fichier .npy trouvé dans {DATA_DIR}"
    print(f"{len(files)} matrices détectées dans {DATA_DIR}")

    # --- 2. Sélection du sujet ---
    SUBJECT_IDX = 5
    A_path = files[SUBJECT_IDX]
    A = load_A(A_path)
    subject_name = os.path.splitext(os.path.basename(A_path))[0].replace("A_", "")
    print(f"Sujet sélectionné : {subject_name}")

    # --- 3. Sélection du nombre de clusters ---
    Q_list = [2, 3, 4, 5, 6]
    best, all_results = model_selection_over_Q(A, Q_list, subject_name=subject_name)

    # --- 4. Résumé du meilleur modèle ---
    print("\n=== Meilleur modèle ===")
    print(f"Q optimal (AIC) : {best['Q']}")
    print(f"AIC : {best['AIC']:.2f}   |   BIC : {best['BIC']:.2f}   |   ICL : {best['ICL']:.2f}")
    print(f"η shape : {best['eta'].shape}   |   Π shape : {best['Pi'].shape}")


    # --- 5. Partition dure (clusters prédits) ---
    y_hat = hard_clusters_from_eta(best["eta"])

    # --- 6. Recherche d’un fichier y_* pour vérifier s’il s’agit de données synthétiques ---
    y_path = os.path.join(DATA_DIR, os.path.basename(A_path).replace("A_", "y_"))
    if os.path.exists(y_path):
        print("\n=== Données synthétiques détectées ===")
        y = np.load(y_path).astype(int).ravel()
        K_true = np.unique(y).size
        print(f"K vrai : {K_true}   |   Q trouvé : {best['Q']}")

        # --- 7. Création de η* (one-hot de y vrai) ---
        eta_true = np.eye(K_true, dtype=float)[y]

        # --- 8. Évaluation complète ---
        evaluate_clustering(
            y_true=y,
            y_pred=y_hat,
            eta_true=eta_true,
            eta_hat=best["eta"]
        )

        # --- 9. Sauvegarde des assignations (dans results/<subject>/assignments/) ---
        save_node_assignments(
            y_true=y,
            y_pred=y_hat,
            mapping=best_label_permutation(y, y_hat),
            A_path=A_path,
            results_root="results"
        )

    else:
        # --- Cas réel : pas de vérité terrain ---
        print("\n=== Données réelles ===")
        print("Aucune vérité terrain disponible.")
        print("Comptages par cluster prédits :", cluster_counts(y_hat))


if __name__ == "__main__":
    main()
