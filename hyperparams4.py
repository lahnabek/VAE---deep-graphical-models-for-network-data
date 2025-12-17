"""
hyperparams.py

Benchmark robuste des hyperparamètres pour Deep LPBM + GCNEncoder
sur données synthétiques assortatives / disassortatives.

Fonctionnalités :
    - Reprise automatique après crash (CSV checkpoint)
    - Aucun stockage en mémoire inutilisé
    - Évaluation complète pour chaque graphe + hyperparam
    - Skip automatique des combinaisons déjà évaluées
"""

import os
import time
import gc
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from deep_lpbm import (
    model_selection_over_Q,
    best_label_permutation,
    global_clustering_scores,
    H_partial_memberships_score,
)

# =====================================================================
# 1) CONFIG
# =====================================================================

RESULT_CSV = "results/hyperparams/benchmark_progress.csv"

os.makedirs("results/hyperparams", exist_ok=True)


# =====================================================================
# 2) Chargement d'un graphe synthétique
# =====================================================================

def load_synthetic_dataset(base_dir, idx):
    A = np.load(os.path.join(base_dir, f"A_{idx:03d}.npy"))
    y = np.load(os.path.join(base_dir, f"y_{idx:03d}.npy"))
    eta = np.load(os.path.join(base_dir, f"eta_{idx:03d}.npy"))
    Pi = np.load(os.path.join(base_dir, f"Pi_{idx:03d}.npy"))
    return A, y, eta, Pi


# =====================================================================
# 3) Vérification si une ligne a déjà été calculée
# =====================================================================

def already_done(csv_path, graph_idx, Q, hidden, lr, neg_ratio):
    if not os.path.exists(csv_path):
        return False

    df = pd.read_csv(csv_path)

    mask = (
        (df["graph_idx"] == graph_idx) &
        (df["Q_tested"] == Q) &
        (df["hidden"] == str(hidden)) &
        (df["lr"] == lr) &
        (df["neg_ratio"] == neg_ratio)
    )
    return mask.any()


# =====================================================================
# 4) Ajout d'une ligne au CSV (append sécurisé)
# =====================================================================

def save_row(row):
    df = pd.DataFrame([row])
    df.to_csv(
        RESULT_CSV,
        mode='a',
        header=not os.path.exists(RESULT_CSV),
        index=False
    )


def apply_permutation_safe(y_pred, perm):
    y_new = []
    for k in y_pred:
        if k in perm:
            y_new.append(perm[k])
        else:
            # cluster fantôme → on mappe vers nearest existing cluster
            y_new.append(perm[min(perm.keys())])  # par défaut
    return np.array(y_new)

# =====================================================================
# 5) Évaluation d’un modèle sur un Q et un graphe
# =====================================================================

def evaluate_one_Q(A, y_true, eta_true, Pi_true,
                   Q, hidden, lr, neg_ratio):

    best, all_fits = model_selection_over_Q(
        A=A,
        Q_list=[Q],
        subject_name=f"temp_Q{Q}",
        negetive_sampling=True,
        neg_ratio=neg_ratio,
        hidden_override=hidden,
        lr_override=lr
    )

    fit = all_fits[0]  # unique car Q_list=[Q]

    # Hard clustering
    eta_pred = fit["eta"]
    Pi_pred = fit["Pi"]
    y_pred = eta_pred.argmax(axis=1)

    # Alignement
    perm = best_label_permutation(y_true, y_pred)
    y_pred_aligned = apply_permutation_safe(y_pred, perm)

    # Scores globaux
    ari, nmi = global_clustering_scores(y_true, y_pred_aligned)

    # Distances soft
    eta_H = H_partial_memberships_score(eta_true, eta_pred)

    # Distance Pi (si même forme)
    Pi_H = np.mean(np.abs(Pi_true - Pi_pred)) if Pi_true.shape == Pi_pred.shape else np.nan

    return {
        "eta_pred": eta_pred,
        "Pi_pred": Pi_pred,
        "ari": ari,
        "nmi": nmi,
        "eta_H": eta_H,
        "Pi_H": Pi_H,
        "ELBO": fit["elbo"],
        "AIC": fit["AIC"],
        "BIC": fit["BIC"],
        "ICL": fit["ICL"],
    }


# =====================================================================
# 6) Benchmark principal : 1 graphe → K hyperparam tests
# =====================================================================

def benchmark_graph(base_dir, graph_idx,
                    Q_list, hidden_list, lr_list, neg_list):

    print(f"\n=== GRAPH #{graph_idx} ===")

    # Chargement GT
    A, y_true, eta_true, Pi_true = load_synthetic_dataset(base_dir, graph_idx)
    Q_true = eta_true.shape[1]

    for hidden in hidden_list:
        for lr in lr_list:
            for neg_ratio in neg_list:
                for Q in Q_list:

                    if already_done(RESULT_CSV, graph_idx, Q, hidden, lr, neg_ratio):
                        print(f"SKIP: graph={graph_idx} Q={Q} hidden={hidden} lr={lr} neg={neg_ratio}")
                        continue

                    print(f"\n→ RUN graph={graph_idx}, Q={Q}, hidden={hidden}, lr={lr}, neg={neg_ratio}")

                    t0 = time.time()

                    out = evaluate_one_Q(
                        A, y_true, eta_true, Pi_true,
                        Q, hidden, lr, neg_ratio
                    )

                    # Enregistrer résultat
                    row = {
                        "graph_idx": graph_idx,
                        "Q_true": Q_true,
                        "Q_tested": Q,
                        "hidden": str(hidden),
                        "lr": lr,
                        "neg_ratio": neg_ratio,

                        "Q_selected": Q,   # model_selection_over_Q(Q_list=[Q]) retourne forcément Q

                        "ELBO": out["ELBO"],
                        "AIC": out["AIC"],
                        "BIC": out["BIC"],
                        "ICL": out["ICL"],

                        "ARI": out["ari"],
                        "NMI": out["nmi"],
                        "eta_H": out["eta_H"],
                        "Pi_H": out["Pi_H"],

                        "time_sec": time.time() - t0
                    }

                    save_row(row)

                    # Nettoyage mémoire
                    plt.close('all')
                    gc.collect()
                    torch.cuda.empty_cache()

                    print(f"✔ DONE in {row['time_sec']:.2f}s")

    print(f"=== END GRAPH {graph_idx} ===")


# =====================================================================
# 7) Script principal
# =====================================================================

if __name__ == "__main__":

    DATA_DIR = "data_synthetic/assortative"
    GRAPH_RANGE = range(0, 10)  # 10 graphes synthétiques

    Q_list = [3, 4, 5, 6]
    hidden_list = [[16], [32], [64]]
    lr_list = [5e-3, 1e-2]
    neg_list = [5]

    for idx in GRAPH_RANGE:
        benchmark_graph(DATA_DIR, idx, Q_list, hidden_list, lr_list, neg_list)

    print("\n==== BENCHMARK COMPLETED ====\n")
