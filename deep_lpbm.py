# deep_lpbm_minimal.py
# Squelette minimal pour répliquer le modèle à partir de matrices .npy déjà prêtes.
# Pas de classes, scikit-learn quand utile, et TODO à compléter.

import os
import glob
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import torch
import torch.nn.functional as F

from class_GCN import *
# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

DATA_DIR = "data_numpy_synthetic"  
RANDOM_STATE = 42
HIDDEN_LAYERS_GCN = 30
MAX_EPOCHS_INIT = 100     # Phase init encodeur (Algorithme 1)
MAX_EPOCHS = 200          # Phase estimation (Algorithme 1)
LEARNING_RATE = 1e-2      # Adam lr=0.01 dans l'article
NUM_SEEDS = 10            # on garde le meilleur ELBO

# ---------------------------------------------------------------------
# PRÉTRAITEMENT (init k-means etc.)
# ---------------------------------------------------------------------
def kmeans_initial_memberships(A, Q):
    """
    - A : NxN binary and symetric matrix
    - Q : number of clusters
    Initialisation douce des appartenances par K-Means sur A (ou ses features simples).
    Rappel article : Initialisation encodeur (Algorithme 1, Sec. 4.2 / App. B.2).
    """
    km = KMeans(n_clusters=Q, n_init="auto", random_state=RANDOM_STATE)
    labels = km.fit_predict(A)  
    N = A.shape[0]
    C = np.zeros((N, Q))
    C[np.arange(N), labels] = 1.0 # one-hot encoding
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

def init_encoder_phase(A, Q, params):
    """
    - A : N×N binaire symétrique ; Q dans params
    Init GNN : µ≈z0 (depuis C_KMeans) et logσ²≈log(0.01). Algorithme 1 / App. B.2.
    """
    device   = torch.device(params.get("device", "cpu"))
    hidden   = int(params.get("hidden", HIDDEN_LAYERS_GCN))
    init_lr  = float(params.get("init_lr", LEARNING_RATE))
    seed     = int(params.get("seed", RANDOM_STATE))
    tau      = float(params.get("init_tau", 1e-3))  # lissage pour éviter log(0)

    torch.manual_seed(seed)
    N = A.shape[0]

    # --- Modèle GCN (créé une fois et mémorisé dans params)
    gcn = params.get("gcn")
    if gcn is None:
        gcn = GCN(in_feats=1, hidden=hidden, out_mu=Q-1, out_lv=1).to(device)
        params["gcn"] = gcn

    # --- Optimiseur pour la phase d'init (Adam lr=0.01 dans l’article)
    opt = params.get("opt_init")
    if opt is None:
        opt = torch.optim.Adam(gcn.parameters(), lr=init_lr)
        params["opt_init"] = opt

    # --- Tenseurs d'entrée (fixes dans la boucle)
    A_t = torch.tensor(A, dtype=torch.float32, device=device)
    X   = torch.ones((N, 1), dtype=torch.float32, device=device)

    # --- Cibles d'init (C_KMeans → inverse softmax Sec. 2 → z0, et logσ²*≈log(0.01))
    C = kmeans_initial_memberships(A, Q)                         # N×Q (numpy)
    z0_np   = np.log((C[:, :Q-1] + tau) / (C[:, [Q-1]] + tau))   # N×(Q−1) inverse softmax
    z0      = torch.tensor(z0_np, dtype=torch.float32, device=device)
    lv_star = torch.full((N, 1), np.log(0.01), dtype=torch.float32, device=device) # cible pour la variance

    last_loss = None
    gcn.train()
    for _ in range(MAX_EPOCHS_INIT):
        opt.zero_grad()
        mu, logvar = gcn(A_t, X)                  #encoder # µ: N×(Q−1), logσ²: N×1
        loss_mu = F.mse_loss(mu, z0)
        loss_lv = F.mse_loss(logvar, lv_star)
        loss = loss_mu + loss_lv                  # ℓ(µ,σ; Z0) de l’Alg. 1
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())

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

def train_deep_lpbm(A, Q, seed=RANDOM_STATE):
    """
    - A : N×N binaire symétrique ; Q : nb clusters
    Entraînement conjoint (Algorithme 1, Sec. 4.2) : encodeur GCN + Π~ (libre) → max ELBO.
    Garde le meilleur seed (ELBO).
    """
    device = torch.device("cpu")
    A_t = torch.tensor(A, dtype=torch.float32, device=device)

    best = {"elbo": -np.inf, "eta": None, "Pi": None}
    for s in range(NUM_SEEDS):
        # --- 0) params / modèles / optims
        params = {"Q": Q, "seed": seed + s, "device": device, "hidden": HIDDEN_LAYERS_GCN, "init_lr": LEARNING_RATE}
        # GCN créé dans l'init
        init_encoder_phase(A, Q, params)             # aligne µ≈Z0 et logσ²≈log(0.01)
        gcn = params["gcn"]

        # Π~ en torch (optimisé conjointement)
        Pi_tilde = torch.zeros((Q, Q), dtype=torch.float32, device=device, requires_grad=True)

        # Optimiseur principal : Adam sur (φ, Π~)
        opt = torch.optim.Adam(list(gcn.parameters()) + [Pi_tilde], lr=LEARNING_RATE)

        # --- 1) boucle estimation conjointe
        gcn.train()
        last_elbo = None
        for _ in range(MAX_EPOCHS):
            opt.zero_grad()

            # encodeur → µ, logσ²
            X = torch.ones((A_t.size(0), 1), dtype=torch.float32, device=device)
            mu, logvar = gcn(A_t, X) # encoder

            # réparamétrisation (1-sample MC)
            z = reparameterize(mu, logvar)   # N×(Q−1)

            # Π = f(Π~)
            Pi = unconstrained_Pi_to_Pi(Pi_tilde)

            # Decodeur
            P = decoder_prob(z, Pi)

            # ELBO
            elbo = elbo_stub(A_t, P, mu, logvar)
            loss = -elbo
            loss.backward()
            opt.step()

            last_elbo = float(elbo.detach().cpu())

        # --- 2) récupérer η, Π et comparer
        with torch.no_grad():
            X = torch.ones((A_t.size(0), 1), dtype=torch.float32, device=device)
            mu, logvar = gcn(A_t, X)
            z = mu                                       # à l’éval, on peut prendre z=µ
            eta = softmax_logits_to_eta(z).cpu().numpy()
            Pi = unconstrained_Pi_to_Pi(Pi_tilde).cpu().numpy()

        if last_elbo > best["elbo"]:
            best["elbo"] = last_elbo
            best["eta"]  = eta
            best["Pi"]   = Pi

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

def model_selection_over_Q(A, Q_list):
    """
    Essaie plusieurs Q et renvoie le meilleur selon AIC (comme recommandé).
    Rappel article : AIC > BIC/ICL pour Deep LPBM (Sec. 5.2).
    """
    results = []
    for Q in Q_list:
        fit = train_deep_lpbm(A, Q)
        AIC, BIC, ICL = compute_AIC_BIC_ICL(A, fit["eta"], fit["Pi"])
        results.append({"Q": Q, "AIC": AIC, "BIC": BIC, "ICL": ICL, **fit})
    best = max(results, key=lambda d: d["AIC"])
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
    files = list_npy_graphs(DATA_DIR)
    assert len(files) > 0, f"Aucun .npy trouvé dans {DATA_DIR}"
    print(f"{len(files)} matrices détectées.")

    # Choisir un sujet (ou itérer)
    SUBJECT_IDX = 0
    A_path = files[SUBJECT_IDX]
    A = load_A(A_path)

    # Sélection Q
    Q_list = [3, 4, 5, 6]
    best, _ = model_selection_over_Q(A, Q_list)

    # Partition dure + affichage
    y_hat = hard_clusters_from_eta(best["eta"])
    print("Best Q (AIC):", best["Q"])
    print("AIC/BIC/ICL:", best["AIC"], best["BIC"], best["ICL"])
    print("η shape / Π shape:", best["eta"].shape, best["Pi"].shape)

    # --- Détection des données synthétiques : on cherche le y correspondant au A sélectionné
    y_path = os.path.join(DATA_DIR, os.path.basename(A_path).replace("A_", "y_"))
    if os.path.exists(y_path):
        # Cas synthétique : on évalue vs vérité terrain
        y = np.load(y_path).astype(int).ravel()
        K_true = int(np.unique(y).size)
        print(f"[synthetic] K vrai = {K_true}  |  Q trouvé = {best['Q']}")

        # H-partial : comparer η* (one-hot de y) à η̂ (soft du modèle)
        N = y.shape[0]
        eta_true = np.eye(K_true, dtype=float)[y]     # (N, K_true)
        H = H_partial_memberships_score(eta_true, best["eta"])
        print("H (partial-membership) :", H)

        
    else:
        print("Aucun y_*.npy détecté → données réelles : métriques supervisées ignorées.")

if __name__ == "__main__":
    main()
