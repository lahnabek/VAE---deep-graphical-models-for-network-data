"""
hyperparams.py

Étude d'hyperparamètres pour Deep LPBM + GCNEncoder
sur un ensemble de graphes synthétiques assortatifs et disassortatifs.

Pour chaque fichier de données :
    - exécute model_selection_over_Q
    - récupère Q sélectionné, scores ARI/NMI, distances η / Π
    - stocke les résultats

Puis moyenne les scores pour chaque combinaison d'hyperparamètres.
"""

import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from deep_lpbm import (
    model_selection_over_Q,
    global_clustering_scores,
    H_partial_memberships_score,
    best_label_permutation,
)


# ==============================================================
# 1) Charger l’ensemble des graphes
# ==============================================================

def load_all_synthetic_graphs(base_dir, n_graphs=10):
    """
    Charge A, y, eta*, Pi* pour une liste de graphes synthétiques.
    base_dir : dossier "assortative" ou "disassortative"
    """
    A_list, y_list, eta_list, Pi_list = [], [], [], []

    for idx in range(0, n_graphs):
        A_path   = os.path.join(base_dir, f"A_{idx:03d}.npy")
        y_path   = os.path.join(base_dir, f"y_{idx:03d}.npy")
        eta_path = os.path.join(base_dir, f"eta_{idx:03d}.npy")
        Pi_path  = os.path.join(base_dir, f"Pi_{idx:03d}.npy")

        print(f"Chargement : {A_path}")

        A_list.append(np.load(A_path))
        y_list.append(np.load(y_path))
        eta_list.append(np.load(eta_path))
        Pi_list.append(np.load(Pi_path))

    return A_list, y_list, eta_list, Pi_list



# ==============================================================
# 2) Évaluer une combinaison d’hyperparamètres SUR UN GRAPHE
# ==============================================================

def evaluate_hparams_on_graph(A, y_true, eta_true, Pi_true,
                              Q_list, hidden, lr, neg_ratio,
                              use_negative_sampling=True, verbose=False):

    Q_true = eta_true.shape[1]

    # Entraînement + sélection de Q
    t0 = time.time()
    best, _ = model_selection_over_Q(
        A=A,
        Q_list=Q_list,
        subject_name=f"h{hidden}_lr{lr}_neg{neg_ratio}",
        negetive_sampling=use_negative_sampling,
        neg_ratio=neg_ratio if use_negative_sampling else 0,
        hidden_override=hidden,
        lr_override=lr
    )
    t1 = time.time()

    # Résultats du meilleur modèle
    eta_pred = best["eta"]
    Pi_pred  = best["Pi"]
    Q_sel    = best["Q"]
    elbo     = best["elbo"]

    # Partition
    y_pred = eta_pred.argmax(axis=1)

    # Alignement des labels
    perm = best_label_permutation(y_true, y_pred)
    y_pred_aligned = np.array([perm[int(k)] for k in y_pred])

    # Scores ARI/NMI
    ari, nmi = global_clustering_scores(y_true, y_pred_aligned)

    # Hellinger sur η
    eta_H = H_partial_memberships_score(eta_true, eta_pred)

    # Distance Π
    Pi_H = np.mean(np.abs(Pi_true - Pi_pred)) \
        if Pi_true.shape == Pi_pred.shape else np.nan

    return {
        "Q_true": Q_true,
        "Q_selected": Q_sel,
        "correct_Q": int(Q_sel == Q_true),
        "ELBO": elbo,
        "AIC": best["AIC"],
        "BIC": best["BIC"],
        "ICL": best["ICL"],
        "ARI": ari,
        "NMI": nmi,
        "eta_H": eta_H,
        "Pi_H": Pi_H,
        "time_sec": t1 - t0,
    }



# ==============================================================
# 3) Grille d'hyperparamètres sur TOUS les graphes
# ==============================================================

def run_hyperparam_search_full(
        A_list, y_list, eta_list, Pi_list,
        Q_list, hidden_list, lr_list, neg_ratio_list,
        use_negative_sampling=True, verbose=True
    ):

    all_results = []

    for hidden in hidden_list:
        for lr in lr_list:
            for neg_ratio in neg_ratio_list:

                if verbose:
                    print("\n===========================================")
                    print(f"Hyperparams : hidden={hidden}  lr={lr}  neg_ratio={neg_ratio}")
                    print("===========================================")

                # Stocker les résultats par graphe
                per_graph = []

                for i, (A, y, eta, Pi) in enumerate(zip(A_list, y_list, eta_list, Pi_list)):
                    
                    print(f"  → Graphe {i+1}/{len(A_list)}")

                    res = evaluate_hparams_on_graph(
                        A, y, eta, Pi,
                        Q_list, hidden, lr, neg_ratio,
                        use_negative_sampling=use_negative_sampling,
                        verbose=verbose
                    )
                    per_graph.append(res)
                    plt.close('all')
                # Moyenne des résultats
                df_graphs = pd.DataFrame(per_graph)
                mean_row = df_graphs.mean(numeric_only=True).to_dict()

                mean_row.update({
                    "hidden": str(hidden),
                    "lr": lr,
                    "neg_ratio": neg_ratio,
                    "n_graphs": len(A_list)
                })

                all_results.append(mean_row)

    return pd.DataFrame(all_results)



# ==============================================================
# 4) Exemple d'utilisation
# ==============================================================

if __name__ == "__main__":

    BASE_DIR = "data_synthetic/assortative"   # ou disassortative
    N_GRAPHS = 10

    print("\n=== Chargement des graphes synthétiques ===")
    A_list, y_list, eta_list, Pi_list = load_all_synthetic_graphs(BASE_DIR, N_GRAPHS)

    # Liste des Q testés automatiquement
    Q_list = [3, 4, 5, 6]

    # Grille d’hyperparamètres
    hidden_list = [[32], [16]]
    lr_list = [1e-2,5e-3]
    neg_ratio_list = [1, 3, 5]

    print("\n=== Lancement du grid search ===")
    df_results = run_hyperparam_search_full(
        A_list, y_list, eta_list, Pi_list,
        Q_list, hidden_list, lr_list, neg_ratio_list,
        use_negative_sampling=True
    )

    print("\n=========== MOYENNES PAR HYPERPARAMÈTRES ===========")
    print(df_results)

    df_results.to_csv("hyperparam_global_results.csv", index=False)
    print("Résultats sauvegardés dans hyperparam_global_results.csv")
