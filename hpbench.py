import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from synthetic_data import generate_synthetic
from deep_lpbm import train_deep_lpbm_GCNEncoder, compute_AIC_BIC_ICL


# -----------------------------
# H-score (partial memberships)
# -----------------------------
def compute_H(eta_true, eta_pred):
    U = eta_true @ eta_true.T
    U_hat = eta_pred @ eta_pred.T
    N = eta_true.shape[0]

    diff = np.abs(U - U_hat)
    tri = np.triu_indices(N, 1)
    diff_sum = diff[tri].sum()

    return np.sqrt(2 * diff_sum / (N * (N - 1)))


# ---------------------------------------------------------------
#  EXPÉRIMENTATION HYPERPARAMÈTRES — DEEPLPBM ONLY
# ---------------------------------------------------------------
def run_deeplpbm_hyper_experiment(
    mode="assortative",
    outdir="synthetic_runs",
    N_list=[12002200, 4400],
    Q=5,
    beta=0.8,
    eps=0.1,
    zetas=np.linspace(0.7, 0.7, 1),
    n_graphs=15,
    hidden_list=[[32]],
    lr_list=[0.005],
    neg_ratio_list=[3],
    Q_list=[5]
):

    os.makedirs("deeplpbm_hyper_results", exist_ok=True)

    # Fichier CSV détaillé (une ligne par exécution)
    detailed_csv = "deeplpbm_hyper_results/deeplpbm_runs_detailed5.csv"
    summary_csv  = "deeplpbm_hyper_results/deeplpbm_hyper_summary5.csv"

    detailed_rows = []   # pour enregistrement complet
    summary_rows  = []   # pour agrégations

    print("\n====================================================")
    print("     DEEP LPBM — COMPARAISON HYPERPARAMÈTRES")
    print("====================================================")

    # Boucle hyperparamètres
    for hidden in hidden_list:
        for lr in lr_list:
            for neg in neg_ratio_list:

                print(f"\n>>> Config: hidden={hidden}, lr={lr}, neg_ratio={neg}")

                ARIs, Hs, NMIs, Times = [], [], [], []

                # Boucle difficulté (zeta)
                for z in zetas:
                    print(f"  - zeta = {z}")
                    for N in N_list:
                        # Plusieurs graphes pour ce z
                        for seed in range(n_graphs):

                            # -----------------------------
                            # Génération du graphe
                            # -----------------------------
                            generate_synthetic(
                                mode=mode,
                                outdir=outdir,
                                n_graphs=1,
                                N=N,
                                Q=Q,
                                beta=beta,
                                eps=eps,
                                zeta=z,
                                seed=seed
                            )
                            for Q_tested in Q_list:
                                # Charger les données
                                A = np.load(f"{outdir}/{mode}/A_000.npy")
                                y_true = np.load(f"{outdir}/{mode}/y_000.npy")
                                eta_true = np.load(f"{outdir}/{mode}/eta_000.npy")

                                # -----------------------------
                                # Appel Deep LPBM (avec timer)
                                # -----------------------------
                                t0 = time.time()
                                fit = train_deep_lpbm_GCNEncoder(
                                    A=A,
                                    Q=Q_tested,
                                    seed=seed,
                                    results_dir=None,
                                    negetive_sampling=False,
                                    neg_ratio=neg,
                                    hidden_override=hidden,
                                    lr_override=lr
                                )
                                t1 = time.time()
                                elapsed = t1 - t0

                                eta_pred = fit["eta"]
                                y_pred = eta_pred.argmax(axis=1)

                                # -----------------------------
                                # Évaluation
                                # -----------------------------
                                AIC, BIC, ICL = compute_AIC_BIC_ICL(A, eta_pred, fit["Pi"])
                                ari = adjusted_rand_score(y_true, y_pred)
                                nmi = normalized_mutual_info_score(y_true, y_pred)
                                H = compute_H(eta_true, eta_pred)

                                ARIs.append(ari)
                                NMIs.append(nmi)
                                Hs.append(H)
                                Times.append(elapsed)

                                # -----------------------------
                                # Sauvegarde détaillée
                                # -----------------------------
                                detailed_rows.append({
                                    "hidden": str(hidden),
                                    "lr": lr,
                                    "neg_ratio": neg,
                                    "zeta": z,
                                    "seed": seed,
                                    "ARI": ari,
                                    "NMI": nmi,
                                    "H": H,
                                    "time_sec": elapsed,
                                    "elbo": fit["elbo"],
                                    "AIC": AIC,
                                    "BIC": BIC,
                                    "ICL": ICL,
                                    "Q_tested": Q_tested,
                                    "N": N
                                })

                # -----------------------------
                # Résumé pour cette config
                # -----------------------------
                summary_rows.append({
                    "hidden": str(hidden),
                    "lr": lr,
                    "neg_ratio": neg,
                    "ARI_mean": np.mean(ARIs),
                    "ARI_std": np.std(ARIs),
                    "H_mean": np.mean(Hs),
                    "H_std": np.std(Hs),
                    "NMI_mean": np.mean(NMIs),
                    "NMI_std": np.std(NMIs),
                    "time_mean": np.mean(Times),
                    "time_std": np.std(Times),
                    "n_runs": len(ARIs)
                })

    # Export des deux CSV
    pd.DataFrame(detailed_rows).to_csv(detailed_csv, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    print("\n=== Résultats détaillés sauvegardés dans:", detailed_csv)
    print("=== Résumé sauvegardé dans:", summary_csv)

    return pd.DataFrame(summary_rows)

def plot():

    # -----------------------------
    # 1. Load data
    # -----------------------------
    df = pd.read_csv("deeplpbm_hyper_results/deeplpbm_runs_detailed.csv")

    # Convert "[32]" → 32
    df["hidden"] = df["hidden"].apply(lambda s: int(s.strip("[]")))

    # -----------------------------
    # 2. Group by hidden dimension
    # -----------------------------
    summary = df.groupby("hidden").agg({
        "H": ["mean", "std"],
        "time_sec": ["mean", "std"]
    })

    # flatten MultiIndex columns
    summary.columns = ["H_mean", "H_std", "time_mean", "time_std"]
    summary = summary.reset_index()

    print(summary)

    # -----------------------------
    # 3. H-score plot with error bars
    # -----------------------------
    plt.figure(figsize=(7, 5))
    plt.bar(
        summary["hidden"].astype(str),
        summary["H_mean"],
        yerr=summary["H_std"],
        capsize=8,
        color="skyblue",
        edgecolor="black"
    )
    plt.xlabel("Hidden layer number")
    plt.ylabel("H score (lower is better)")
    plt.title("Partial Membership Error vs Hidden Layer Number")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # 4. Runtime plot with error bars
    # -----------------------------
    plt.figure(figsize=(7, 5))
    plt.bar(
        summary["hidden"].astype(str),
        summary["time_mean"],
        yerr=summary["time_std"],
        capsize=8,
        color="salmon",
        edgecolor="black"
    )
    plt.xlabel("Hidden layer number")
    plt.ylabel("Mean runtime (seconds)")
    plt.title("Runtime vs Hidden Layer Number")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

def plotTime():

    # -----------------------------
    # 1. Load data
    # -----------------------------
    df = pd.read_csv("deeplpbm_hyper_results/deeplpbm_runs_detailed2.csv")


    # -----------------------------
    # 2. Group by hidden dimension
    # -----------------------------
    summary = df.groupby("lr").agg({
        "H": ["mean", "std"],
        "time_sec": ["mean", "std"]
    })

    # flatten MultiIndex columns
    summary.columns = ["H_mean", "H_std", "time_mean", "time_std"]
    summary = summary.reset_index()

    print(summary)

    # -----------------------------
    # 3. H-score plot with error bars
    # -----------------------------
    plt.figure(figsize=(7, 5))
    plt.bar(
        summary["lr"].astype(str),
        summary["H_mean"],
        yerr=summary["H_std"],
        capsize=8,
        color="skyblue",
        edgecolor="black"
    )
    plt.xlabel("Learning rate")
    plt.ylabel("H score (lower is better)")
    plt.title("Partial Membership Error vs Learning rate")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # 4. Runtime plot with error bars
    # -----------------------------
    plt.figure(figsize=(7, 5))
    plt.bar(
        summary["lr"].astype(str),
        summary["time_mean"],
        yerr=summary["time_std"],
        capsize=8,
        color="salmon",
        edgecolor="black"
    )
    plt.xlabel("Learning rate")
    plt.ylabel("Mean runtime (seconds)")
    plt.title("Runtime vs Learning rate")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

def plotAIC():

    # -----------------------------
    # 1. Load data
    # -----------------------------
    df = pd.read_csv("deeplpbm_hyper_results/deeplpbm_runs_detailed3.csv")


    # -----------------------------
    # 2. Group by hidden dimension
    # -----------------------------
    summary = df.groupby("Q_tested").agg({
        "AIC": ["mean", "std"],
        "time_sec": ["mean", "std"]
    })

    # flatten MultiIndex columns
    summary.columns = ["AIC_mean", "AIC_std", "time_mean", "time_std"]
    summary = summary.reset_index()

    print(summary)

    # -----------------------------
    # 3. H-score plot with error bars
    # -----------------------------
    plt.figure(figsize=(7, 5))
    plt.bar(
        summary["Q_tested"].astype(str),
        summary["AIC_mean"],
        yerr=summary["AIC_std"],
        capsize=8,
        color="skyblue",
        edgecolor="black"
    )
    plt.xlabel("Q tested")
    plt.ylabel("AIC score (higher is better)")
    plt.title("AIC vs Q tested")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # 4. Runtime plot with error bars
    # -----------------------------
    plt.figure(figsize=(7, 5))
    plt.bar(
        summary["Q_tested"].astype(str),
        summary["time_mean"],
        yerr=summary["time_std"],
        capsize=8,
        color="salmon",
        edgecolor="black"
    )
    plt.xlabel("Q tested")
    plt.ylabel("Mean runtime (seconds)")
    plt.title("Runtime vs Q tested")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot2():

    # -----------------------------
    # 1. Charger les données
    # -----------------------------
    df = pd.read_csv("deeplpbm_hyper_results/deeplpbm_runs_detailed.csv")

    # Convertir la colonne hidden "[32]" → 32
    df["hidden"] = df["hidden"].apply(lambda s: int(s.strip("[]")))

    # -----------------------------
    # 2. Regrouper par hidden
    # -----------------------------
    summary = df.groupby("hidden").agg({
        "H": "mean",
        "time_sec": "mean"
    }).reset_index()

    print(summary)

    # -----------------------------
    # 3. Plot du score H
    # -----------------------------
    plt.figure(figsize=(7,5))
    plt.bar(summary["hidden"].astype(str), summary["H"], color="skyblue")
    plt.xlabel("Hidden dimension size")
    plt.ylabel("H score (lower is better)")
    plt.title("Partial Membership Error vs Hidden Dimension")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # 4. Plot du temps
    # -----------------------------
    plt.figure(figsize=(7,5))
    plt.bar(summary["hidden"].astype(str), summary["time_sec"], color="salmon")
    plt.xlabel("Hidden dimension size")
    plt.ylabel("Mean runtime (seconds)")
    plt.title("Runtime vs Hidden Dimension")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

def mainARi():
    df = pd.read_csv("deeplpbm_hyper_results/deeplpbm_runs_detailed3.csv")

    group_cols = ["seed"]

    # Sélectionner le Q avec ARI maximal
    selected_AIC = df.groupby(group_cols).apply(
        lambda g: pd.Series({
            "Q_true": 5,
            "Q_AIC": g.loc[g["AIC"].idxmax(), "Q_tested"],
            "AIC_best": g["AIC"].max()
        })
    ).reset_index()

    # Score de réussite : Q_ARI == Q_true
    selected_AIC["AIC_correct"] = (selected_AIC["Q_AIC"] == selected_AIC["Q_true"]).astype(int)

    print("\n📌 Taux de réussite AIC pour la sélection du bon Q :")
    print(selected_AIC["AIC_correct"].mean())

    mean_val_ari = selected_AIC["AIC_correct"].mean()
    ########
    selected_BIC = df.groupby(group_cols).apply(
        lambda g: pd.Series({
            "Q_true": 5,
            "Q_BIC": g.loc[g["BIC"].idxmin(), "Q_tested"],
            "BIC_best": g["BIC"].min()
        })
    ).reset_index()

    # Score de réussite : Q_ARI == Q_true
    selected_BIC["BIC_correct"] = (selected_BIC["Q_BIC"] == selected_BIC["Q_true"]).astype(int)

    print("\n📌 Taux de réussite BIC pour la sélection du bon Q :")
    print(selected_BIC["BIC_correct"].mean())

    mean_val_bic = selected_BIC["BIC_correct"].mean()
    ####

    selected_ICL = df.groupby(group_cols).apply(
        lambda g: pd.Series({
            "Q_true": 5,
            "Q_ICL": g.loc[g["ICL"].idxmax(), "Q_tested"],
            "ICL_best": g["ICL"].max()
        })
    ).reset_index()

    # Score de réussite : Q_ARI == Q_true
    selected_ICL["ICL_correct"] = (selected_ICL["Q_ICL"] == selected_ICL["Q_true"]).astype(int)

    print("\n📌 Taux de réussite ARI pour la sélection du bon Q :")
    print(selected_ICL["ICL_correct"].mean())

    mean_val_icl = selected_ICL["ICL_correct"].mean()

    plt.figure(figsize=(4,4))
    plt.bar(["AIC","BIC", "ICL"], [mean_val_ari, mean_val_bic, mean_val_icl])
    plt.ylim(0,1)
    plt.ylabel("Proportion of correct Q")
    plt.title("Rate of right Q selected via AIC vs BIC vs ICL")
    plt.grid(axis="y")
    plt.show()

def plotHvN():

    # -----------------------------
    # 1. Load data
    # -----------------------------
    df = pd.read_csv("deeplpbm_hyper_results/deeplpbm_runs_detailed4.csv")


    # -----------------------------
    # 2. Group by hidden dimension
    # -----------------------------
    summary = df.groupby("N").agg({
        "H": ["mean", "std"],
        "time_sec": ["mean", "std"]
    })

    # flatten MultiIndex columns
    summary.columns = ["H_mean", "H_std", "time_mean", "time_std"]
    summary = summary.reset_index()

    print(summary)

    # -----------------------------
    # 3. H-score plot with error bars
    # -----------------------------
    plt.figure(figsize=(7, 5))
    plt.bar(
        summary["N"].astype(str),
        summary["H_mean"],
        yerr=summary["H_std"],
        capsize=8,
        color="skyblue",
        edgecolor="black"
    )
    plt.xlabel("N")
    plt.ylabel("H score (lower is better)")
    plt.title("H vs N")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # 4. Runtime plot with error bars
    # -----------------------------
    plt.figure(figsize=(7, 5))
    plt.bar(
        summary["N"].astype(str),
        summary["time_mean"],
        yerr=summary["time_std"],
        capsize=8,
        color="salmon",
        edgecolor="black"
    )
    plt.xlabel("N")
    plt.ylabel("Mean runtime (seconds)")
    plt.title("Runtime vs N")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------
#  EXÉCUTION DIRECTE
# ---------------------------------------------------------------
if __name__ == "__main__":
    #run_deeplpbm_hyper_experiment()
    plotHvN()
    #mainARi()
    #plotTime()