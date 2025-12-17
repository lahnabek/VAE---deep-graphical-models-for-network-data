"""
hyperparams.py

Tests d'hyperparamètres du modèle Deep LPBM + GCNEncoder
sur un jeu de données SYNTHÉTIQUES.

Utilise :
    model_selection_over_Q
    global_clustering_scores
    H_partial_memberships_score
"""

import time
import os
import numpy as np
import pandas as pd

from deep_lpbm import (
    model_selection_over_Q,
    global_clustering_scores,
    H_partial_memberships_score,
    best_label_permutation,
)


# ==============================================================
# 1) Grille d'hyperparamètres utilisant model_selection_over_Q
# ==============================================================

def run_hyperparam_search_synthetic(
        A, y_true, eta_true, Pi_true,
        Q_list,
        hidden_list,
        lr_list,
        neg_ratio_list,
        use_negative_sampling=True,
        seed=0,
        verbose=True
    ):
    """
    Pour chaque combinaison d'hyperparamètres :
      - lance model_selection_over_Q
      - récupère le meilleur Q
      - calcule ARI / NMI / dist_eta / dist_Pi
      - enregistre résultats dans un DataFrame
    """
    Q_true = eta_true.shape[1]
    results = []
    results_root = os.path.join("results", "hyperparams")
    os.makedirs(results_root, exist_ok=True)

    for hidden in hidden_list:
        for lr in lr_list:
            for neg_ratio in (neg_ratio_list if use_negative_sampling else [None]):

                if verbose:
                    print("\n===========================================")
                    print(f"Test configuration :")
                    print(f"   hidden     = {hidden}")
                    print(f"   lr         = {lr}")
                    print(f"   neg_ratio  = {neg_ratio}")
                    print("===========================================")

                t0 = time.time()

                # Appel au sélecteur de modèle
                best, all_results = model_selection_over_Q(
                    A=A,
                    Q_list=Q_list,
                    subject_name=f"h{hidden}_lr{lr}_neg{neg_ratio}",
                    negetive_sampling=use_negative_sampling,
                    neg_ratio=neg_ratio if use_negative_sampling else 0,
                    hidden_override=hidden,
                    lr_override=lr
                )

                t1 = time.time()

                # Extraire les résultats du meilleur modèle
                eta_pred = best["eta"]
                Pi_pred = best["Pi"]
                Q_best = best["Q"]
                elbo = best["elbo"]

                # Partition prédite
                y_pred = eta_pred.argmax(axis=1)

                # Alignement des labels
                perm = best_label_permutation(y_true, y_pred)
                y_pred_aligned = np.array([perm[int(k)] for k in y_pred])

                # Scores ARI / NMI
                ari, nmi = global_clustering_scores(y_true, y_pred_aligned)

                # Distances η
                eta_H = H_partial_memberships_score(eta_true, eta_pred)
                if Pi_true.shape == Pi_pred.shape:
                    Pi_H = np.mean(np.abs(Pi_true - Pi_pred))
                else:
                    Pi_H = np.nan
                

                # Enregistrement
                results.append({
                    "hidden": str(hidden),
                    "lr": lr,
                    "neg_ratio": neg_ratio,
                    "Q_true": Q_true,
                    "Q_selected": Q_best,
                    "ELBO": elbo,
                    "AIC": best["AIC"],
                    "BIC": best["BIC"],
                    "ICL": best["ICL"],
                    "ARI": ari,
                    "NMI": nmi,
                    "eta_H": eta_H,
                    "Pi_H": Pi_H,
                    "time_sec": t1 - t0
                })

    return pd.DataFrame(results)


# ==============================================================
# 2) Chargement d'un dataset synthétique
# ==============================================================

def load_synthetic_dataset(base_dir, subject_idx):
    """Charge A, y, eta*, Pi* pour un dataset synthétique."""

    A_path   = os.path.join(base_dir, f"A_{subject_idx:03d}.npy")
    y_path   = os.path.join(base_dir, f"y_{subject_idx:03d}.npy")
    eta_path = os.path.join(base_dir, f"eta_{subject_idx:03d}.npy")
    Pi_path  = os.path.join(base_dir, f"Pi_{subject_idx:03d}.npy")

    print("Chargement des données synthétiques :")
    print(" ", A_path)
    print(" ", y_path)
    print(" ", eta_path)
    print(" ", Pi_path)

    return (
        np.load(A_path),
        np.load(y_path),
        np.load(eta_path),
        np.load(Pi_path)
    )


# ==============================================================
# 3) Exemple d'utilisation
# ==============================================================

if __name__ == "__main__":

    DATA_DIR = "data_synthetic/assortative"   # ou "data_synthetic/disassortative"
    SUBJECT_IDX = 2

    # Chargement de A, y, eta*, Pi*
    A, y_true, eta_true, Pi_true = load_synthetic_dataset(DATA_DIR, SUBJECT_IDX)

    # Liste des Q à tester automatiquement
    Q_list = [3, 4, 5, 6]

    # Grille d'hyperparamètres
    hidden_list = [[16], [32], [64]]
    lr_list = [5e-3, 1e-2]
    neg_ratio_list = [1, 3, 5]

    df = run_hyperparam_search_synthetic(
        A=A,
        y_true=y_true,
        eta_true=eta_true,
        Pi_true=Pi_true,
        Q_list=Q_list,
        hidden_list=hidden_list,
        lr_list=lr_list,
        neg_ratio_list=neg_ratio_list,
        use_negative_sampling=True,
        seed=0,
    )

    print("\n=========== RÉSULTATS HYPERPARAMÈTRES ===========")
    print(df)

    df.to_csv(f"hyperparam_synthetic_results{SUBJECT_IDX}.csv", index=False)
    print("Résultats sauvegardés dans hyperparam_synthetic_results.csv")
