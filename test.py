import numpy as np
import matplotlib.pyplot as plt
A = np.load("dataset_numpy_threshold/A_subject_000.npy")


def check_adjacency_matrix(A: np.ndarray, n_nodes: int = 180, verbose: bool = True) -> bool:
    """
    Vérifie la validité structurelle d'une matrice d'adjacence A.
    - dimensions correctes
    - symétrie
    - diagonale nulle
    - valeurs binaires (0 ou 1)
    - connexité (optionnelle)
    """

    ok = True

    # 1️⃣ Dimensions
    if A.shape != (n_nodes, n_nodes):
        print(f"❌ Mauvaise taille: {A.shape}, attendu {(n_nodes, n_nodes)}")
        ok = False
    elif verbose:
        print("✅ Dimensions correctes")

    # 2️⃣ Symétrie
    if not np.allclose(A, A.T):
        print("❌ La matrice n’est pas symétrique")
        ok = False
    elif verbose:
        print("✅ Symétrie OK")

    # 3️⃣ Diagonale nulle
    if np.any(np.diag(A) != 0):
        print("❌ Diagonale non nulle (boucles présentes)")
        ok = False
    elif verbose:
        print("✅ Diagonale nulle")

    # 4️⃣ Valeurs binaires
    unique_vals = np.unique(A)
    if not np.all(np.isin(unique_vals, [0, 1])):
        print(f"❌ Valeurs inattendues : {unique_vals}")
        ok = False
    elif verbose:
        print("✅ Valeurs binaires")

    # 5️⃣ Optionnel : connexité (si NetworkX dispo)
    try:
        import networkx as nx
        G = nx.from_numpy_array(A)
        if not nx.is_connected(G):
            print("⚠️ Graphe non connexe : certaines régions ne sont pas reliées")
        elif verbose:
            print("✅ Graphe connexe")
    except ImportError:
        pass

    if ok and verbose:
        print("🎉 La matrice A est bien structurée.")
    return ok

ok = check_adjacency_matrix(A)


