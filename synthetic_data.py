# synthetic_data.py
import os
import json
import numpy as np
from pathlib import Path


def _build_connectivity_matrix(mode, Q, beta, eps):
    """Construit Π selon les 3 structures de l’article (communities, disassortative, hub)."""

    Pi = np.zeros((Q, Q))

    if mode == "assortative":
        for i in range(Q):
            for j in range(Q):
                Pi[i, j] = beta if i == j else eps

    elif mode == "disassortative":
        for i in range(Q):
            for j in range(Q):
                Pi[i, j] = eps if i == j else beta

    elif mode == "hub":
        Pi[:] = eps
        hub_cluster = 0                           # cluster choisi comme hub
        Pi[hub_cluster, :] = beta
        Pi[:, hub_cluster] = beta
        for q in range(1, Q):
            Pi[q, q] = beta                       # les autres restent "communities-like"
    else:
        raise ValueError(f"Unknown mode {mode}")

    # assure symétrie
    Pi = 0.5 * (Pi + Pi.T)
    return Pi


def _build_partial_memberships(N, Q, zeta):
    """Construit η⋆ = ζ·onehot + (1-ζ)·uniform comme dans la section 5.1."""
    q_per_block = N // Q
    labels = np.repeat(np.arange(Q), q_per_block)
    if len(labels) < N:
        labels = np.concatenate([labels, np.full(N - len(labels), Q-1)])

    eta_onehot = np.zeros((N, Q))
    for i, c in enumerate(labels):
        eta_onehot[i, c] = 1

    eta_uniform = np.ones((N, Q)) / Q
    eta_star = zeta * eta_onehot + (1 - zeta) * eta_uniform

    return eta_star, labels


def _sample_graph(P):
    """Tirage Bernoulli(i,j) avec symétrie."""
    N = P.shape[0]
    A = np.zeros((N, N), dtype=np.uint8)

    for i in range(N):
        for j in range(i+1, N):
            Aij = np.random.binomial(1, P[i, j])
            A[i, j] = Aij
            A[j, i] = Aij
    return A


def generate_synthetic(outdir="data_synthetic",
                       mode="disassortative",
                       n_graphs=10,
                       N=200,
                       Q=5,
                       beta=0.3,
                       eps=0.05,
                       zeta=1.0,
                       seed=0):
    """
    Génère des données synthétiques à la Deep LPBM :
    - Matrice Π (Q×Q)
    - Membres latents η⋆
    - Probabilités P = η⋆ Π η⋆ᵀ
    - Matrice d’adjacence A

    Sauvegarde :
      A_XXX.npy, y_XXX.npy, Pi_XXX.npy, metadata.json
    """

    np.random.seed(seed)

    mode_dir = Path(outdir) / mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    # --- meta
    metadata = {
        "mode": mode,
        "N": N,
        "Q": Q,
        "beta": beta,
        "eps": eps,
        "zeta": zeta,
        "n_graphs": n_graphs
    }

    # --- génération de n_graphs
    for k in range(n_graphs):

        # Connectivité Π
        Pi = _build_connectivity_matrix(mode, Q, beta, eps)

        # Membres η⋆
        eta_star, labels = _build_partial_memberships(N, Q, zeta)

        # Matrice de probabilité
        P = eta_star @ Pi @ eta_star.T

        # Matrice d'adjacence
        A = _sample_graph(P)

        # Sauvegardes
        np.save(mode_dir / f"A_{k:03d}.npy", A)
        np.save(mode_dir / f"y_{k:03d}.npy", labels)
        np.save(mode_dir / f"Pi_{k:03d}.npy", Pi)
        np.save(mode_dir / f"eta_{k:03d}.npy", eta_star)

    with open(mode_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return str(mode_dir)


if __name__ == "__main__":
    # Exemple : générer 10 graphes disassortatifs
    generate_synthetic(mode="hub", #assortative, hub, disassortative
                       outdir="data_synthetic",
                       n_graphs=10,
                       N=200,
                       Q=3,
                       beta=0.3,
                       eps=0.05,
                       zeta=1.0, # 1 for hard clustering and in (0,1) for partial
                       seed=0)
