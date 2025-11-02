# synthetic_sbm.py
import os
import json
import numpy as np
import networkx as nx
from pathlib import Path

# Générateur aléatoire éventuellement utile si tu veux tirer toi-même des paramètres.
# (Non utilisé dans les fonctions ci-dessous, car on délègue le hasard à networkx via `seed`.)
rng = np.random.default_rng()


def make_sbm(N: int = 180,
             K: int = 8,
             p_in: float = 0.25,
             p_out: float = 0.02,
             sizes: list | None = None,
             seed: int | None = 0):
    """
    Génère un graphe **SBM** (Stochastic Block Model) *assortaÏif* avec `K` clusters,
    en imposant une probabilité de connexion **intra-cluster** `p_in` et **inter-cluster** `p_out`.

    Paramètres
    ----------
    N : int, par défaut 180
        Nombre total de nœuds. Influence directe sur la taille de la matrice d’adjacence `A` (N x N).
        Plus `N` est grand, plus le graphe peut être dense (si p_in/p_out élevés) et coûteux en mémoire/temps.
    K : int, par défaut 8
        Nombre de communautés (blocs/clusters). Contrôle la granularité : plus `K` est grand, plus
        les communautés sont petites (si `sizes` n’est pas fourni).
    p_in : float, par défaut 0.25
        Probabilité d’arête **à l’intérieur** d’un même cluster. Plus `p_in` est élevé (proche de 1),
        plus chaque bloc est dense (structure “communautaire” marquée).
    p_out : float, par défaut 0.02
        Probabilité d’arête **entre** deux clusters différents. Plus `p_out` est bas (proche de 0),
        plus la séparation entre communautés est nette. Si `p_out` s’approche de `p_in`, la structure de clusters
        devient difficile à détecter.
    sizes : list[int] | None, par défaut None
        Liste des tailles de chaque cluster (longueur `K`) dont la somme doit faire `N`.
        - Si `None`, on répartit `N` équitablement entre les `K` clusters (écart ≤ 1).
        - L’influence de `sizes` est majeure sur l’équilibre/déséquilibre des classes (utile pour tester
          des cas difficiles).
    seed : int | None, par défaut 0
        Graine pseudo-aléatoire passée à `networkx.stochastic_block_model` pour la **reproductibilité**.
        - Fixer un entier ⇒ le même graphe sera régénéré.
        - Mettre `None` ⇒ graphe différent à chaque appel.

    Retours
    -------
    A : np.ndarray, dtype uint8, shape (N, N)
        Matrice d’adjacence **binaire, symétrique, sans boucles** (diagonale nulle), où A[i, j] = 1 s’il y a arête.
    y : np.ndarray, shape (N,)
        Étiquettes de cluster (entiers 0..K-1) alignées avec les indices des nœuds (ordre de construction).

    Exceptions
    ----------
    AssertionError
        - Si `sizes` est fourni et que `sum(sizes) != N`.
        - Si `len(sizes) != K` quand `sizes` est fourni.

    Exemples
    --------
    >>> A, y = make_sbm(N=180, K=8, p_in=0.3, p_out=0.01, seed=42)
    >>> A.shape  # (180, 180)
    (180, 180)
    >>> len(np.unique(y))  # 8 clusters
    8
    """
    # --- Vérifications et préparation des tailles de clusters
    if sizes is None:
        # Répartition quasi uniforme : on distribue le reste r sur les r premiers clusters
        q, r = divmod(N, K)
        sizes = [q + (1 if i < r else 0) for i in range(K)]
    else:
        # Si sizes est fourni, on s'assure qu'il est cohérent
        assert len(sizes) == K, "len(sizes) doit être égal à K."
        assert sum(sizes) == N, "La somme de sizes doit être égale à N."

    # --- Matrice de probas bloc-bloc (K x K)
    # Diagonale = p_in (intra-cluster) ; hors-diagonale = p_out (inter-cluster)
    P = [[p_in if i == j else p_out for j in range(K)] for i in range(K)]

    # --- Génération du graphe SBM non orienté, sans boucles
    # - directed=False ⇒ matrice d’adjacence symétrique
    # - selfloops=False ⇒ pas d’arêtes (i, i)
    G = nx.stochastic_block_model(
        sizes, P, seed=seed, directed=False, selfloops=False
    )

    # --- Conversion en matrice d’adjacence binaire (uint8 pour économiser un peu)
    A = nx.to_numpy_array(G, dtype=np.uint8)

    # --- Génération du vecteur d’étiquettes y (0..K-1) aligné à l’ordre d’empilement des blocs
    y = np.concatenate([[k] * sizes[k] for k in range(K)])

    return A, y


def save_batch(outdir: str = "data_numpy_synthetic",
               n_graphs: int = 10,
               **sbm_kwargs):
    """
    Génère un **lot de graphes** SBM et enregistre pour chacun :
    - `A_XXX.npy` : matrice d’adjacence binaire (N x N)
    - `y_XXX.npy` : labels de cluster (longueur N)
    Et écrit un `metadata.json` avec les principaux hyperparamètres utilisés.

    Paramètres
    ----------
    outdir : str, par défaut "data_numpy_synthetic"
        Dossier de sortie. Sera créé s’il n’existe pas. Influence : organisation du pipeline,
        pratique pour versionner plusieurs jeux synthétiques.
    n_graphs : int, par défaut 10
        Nombre de graphes à générer. Influence : volume de données pour l’entraînement/évaluation.
    **sbm_kwargs :
        Arguments passés **tels quels** à `make_sbm`. Typiquement :
        - N, K, p_in, p_out, sizes, seed
        Astuce :
        - Si tu passes `seed=s`, chaque graphe recevra `seed=s+i` (i = 0..n_graphs-1)
          pour obtenir des graphes distincts mais **reproductibles**.

    Retours
    -------
    outdir : str
        Le chemin du dossier où les fichiers ont été écrits (pratique pour chaîner dans un script).

    Fichiers générés
    ----------------
    - `{outdir}/A_000.npy`, `{outdir}/y_000.npy`, ..., jusqu’à `A_{n_graphs-1:03d}.npy`
    - `{outdir}/metadata.json` contenant :
        {
          "N": <int>,
          "K": <int>,
          "p_in": <float>,
          "p_out": <float>,
          "sizes": <list | null>,
          "n_graphs": <int>
        }

    
    """
    # Crée le dossier s'il n'existe pas (parents=True ⇒ crée toute la hiérarchie si besoin)
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Métadonnées de base (on complétera après le 1er graphe)
    meta = dict(N=None, K=None, p_in=None, p_out=None, sizes=None, n_graphs=n_graphs)

    # Boucle de génération des graphes
    for i in range(n_graphs):
        # Important : on décale la graine à chaque itération pour diversifier les graphes
        # tout en restant déterministe si une `seed` initiale est fournie.
        A, y = make_sbm(**{**sbm_kwargs, "seed": (sbm_kwargs.get("seed", 0) + i)})

        # Enregistrement des matrices / labels
        np.save(os.path.join(outdir, f"A_{i:03d}.npy"), A)
        np.save(os.path.join(outdir, f"y_{i:03d}.npy"), y)

        # Remplissage des méta au premier passage (N, K, etc.)
        if i == 0:
            meta.update(
                N=A.shape[0],
                K=len(np.unique(y)),
                p_in=sbm_kwargs.get("p_in", 0.25),
                p_out=sbm_kwargs.get("p_out", 0.02),
                sizes=[int(s) for s in sbm_kwargs.get("sizes", [])] or None
            )

    # Écriture des métadonnées globales
    with open(os.path.join(outdir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return outdir


if __name__ == "__main__":
    # Exemple "proche article" : 180 régions, ~8 réseaux, clusters quasi équilibrés,
    # forte densité intra (p_in) et faible inter (p_out) ⇒ communautés bien séparées.
    save_batch(outdir="data_numpy_synthetic", n_graphs=10,
               N=50, K=3, p_in=0.99, p_out=0.01, seed=0)
