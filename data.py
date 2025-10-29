"""
Création d'un dataset numpy à partir des graphes HCP-100 (Glasser-180)
----------------------------------------------------------------------
Ce script :
1. charge les edge-lists (graphes binaires par sujet)
2. reconstruit chaque matrice d’adjacence A (180x180)
3. enregistre chaque matrice dans un dossier numpy/
"""

import os
import numpy as np
import pandas as pd

# -------------------------------
# Paramètres de base
# -------------------------------
DATA_FOLDER = "data_human_brain_networks/graphs_spanningtree_180"  # ou graphs_spanningtree_180
OUTPUT_FOLDER = "dataset_numpy_spanningtree"
N_REGIONS = 180

# Créer le dossier de sortie s’il n’existe pas
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------------------------------
# Fonction utilitaire
# -------------------------------
def edge_list_to_matrix(path: str, n_nodes: int = 180) -> np.ndarray:
    """Charge un fichier edge-list et retourne une matrice d’adjacence binaire symétrique A"""
    df = pd.read_csv(path, sep=' ', header=None, names=['u', 'v'])
    df[['u', 'v']] = df[['u', 'v']] - 1  # indices 0-based
    A = np.zeros((n_nodes, n_nodes), dtype=np.uint8)
    for _, row in df.iterrows():
        i, j = int(row['u']), int(row['v'])
        A[i, j] = 1
        A[j, i] = 1
    np.fill_diagonal(A, 0)
    return A

# -------------------------------
#  Traitement principal
# -------------------------------
files = sorted([f for f in os.listdir(DATA_FOLDER) if f.endswith(".txt") or f.endswith(".csv")])
print(f"Chargement de {len(files)} graphes depuis {DATA_FOLDER}")

for idx, filename in enumerate(files):
    file_path = os.path.join(DATA_FOLDER, filename)
    A = edge_list_to_matrix(file_path, n_nodes=N_REGIONS)

    # Enregistrement au format numpy
    save_path = os.path.join(OUTPUT_FOLDER, f"A_subject_{idx:03d}.npy")
    np.save(save_path, A)

    if idx % 10 == 0:
        print(f"Graphe {idx+1}/{len(files)} sauvegardé → {save_path}")

print(f"\n Toutes les matrices A ont été sauvegardées dans {os.path.abspath(OUTPUT_FOLDER)}")

# -------------------------------
#  Vérification d’un exemple
# -------------------------------
sample = np.load(os.path.join(OUTPUT_FOLDER, "A_subject_000.npy"))
print(f"Exemple : matrice shape = {sample.shape}, densité = {sample.sum() / (N_REGIONS**2):.4f}")
