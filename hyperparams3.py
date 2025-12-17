import os
import time
import gc
import numpy as np
import pandas as pd
import torch
from deep_lpbm import (
    model_selection_over_Q,
    global_clustering_scores,
    H_partial_memberships_score,
    best_label_permutation
)
import matplotlib.pyplot as plt

RESULTS_PATH = "results/hyperparams/full_results.csv"

def evaluate_one_graph(A, y_true, eta_true, Pi_true,
                       Q_list, hidden, lr, neg_ratio,
                       use_negative_sampling):

    Q_true = eta_true.shape[1]
    row_results = []

    # Sélection de modèle sur Q_list
    best, all_models = model_selection_over_Q(
        A=A,
        Q_list=Q_list,
        subject_name="temp_eval",
        negetive_sampling=use_negative_sampling,
        neg_ratio=neg_ratio,
        hidden_override=hidden,
        lr_override=lr
    )

    # Pour chaque modèle testé (un par Q)
    for fit in all_models:
        Q = fit["Q"]
        eta_pred = fit["eta"]
        Pi_pred  = fit["Pi"]
        elbo = fit["elbo"]

        # Hard clustering
        y_pred = eta_pred.argmax(axis=1)

        # Alignement
        perm = best_label_permutation(y_true, y_pred)
        y_pred_aligned = np.array([perm[k] for k in y_pred])

        # Scores globaux
        ari, nmi = global_clustering_scores(y_true, y_pred_aligned)

        # Distances
        eta_H = H_partial_memberships_score(eta_true, eta_pred)
        Pi_H = np.mean(np.abs(Pi_true - Pi_pred)) if Pi_true.shape == Pi_pred.shape else np.nan

        row_results.append({
            "Q_true": Q_true,
            "Q_tested": Q,
            "Q_selected": best["Q"],
            "hidden": str(hidden),
            "lr": lr,
            "neg_ratio": neg_ratio,
            "ELBO": elbo,
            "AIC": fit["AIC"],
            "BIC": fit["BIC"],
            "ICL": fit["ICL"],
            "ARI": ari,
            "NMI": nmi,
            "eta_H": eta_H,
            "Pi_H": Pi_H
        })

    return row_results


def benchmark_all_graphs(data_dir, Q_list, hidden_list, lr_list, neg_list,
                         use_negative_sampling=True):

    all_rows = []

    for file in sorted(os.listdir(data_dir)):
        if not file.startswith("A_"):
            continue

        idx = int(file.split("_")[1].split(".")[0])
        print(f"\n=== GRAPH #{idx} ===")

        A = np.load(os.path.join(data_dir, file))
        y = np.load(os.path.join(data_dir, f"y_{idx:03d}.npy"))
        eta = np.load(os.path.join(data_dir, f"eta_{idx:03d}.npy"))
        Pi  = np.load(os.path.join(data_dir, f"Pi_{idx:03d}.npy"))

        for hidden in hidden_list:
            for lr in lr_list:
                for neg_ratio in neg_list:

                    t0 = time.time()

                    results = evaluate_one_graph(
                        A, y, eta, Pi,
                        Q_list, hidden, lr, neg_ratio,
                        use_negative_sampling
                    )

                    # Écriture immédiate dans le CSV
                    df = pd.DataFrame(results)
                    df["graph_idx"] = idx
                    df.to_csv(RESULTS_PATH, mode='a', header=not os.path.exists(RESULTS_PATH), index=False)

                    print(f"Graph {idx} finished in {time.time()-t0:.1f}s")

                    # Libération mémoire
                    plt.close('all')
                    torch.cuda.empty_cache()
                    gc.collect()


if __name__ == "__main__":

    DATA_DIR = "data_synthetic/assortative"
    Q_list = [3,4,5,6]
    hidden_list = [[16],[32],[64]]
    lr_list = [5e-3, 1e-2]
    neg_list = [1,3,5]

    benchmark_all_graphs(DATA_DIR, Q_list, hidden_list, lr_list, neg_list)
    print("Benchmark completed.")