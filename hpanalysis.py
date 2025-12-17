import os
import time
import gc
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import ast

def main():
    df = pd.read_csv("results/hyperparams/benchmark_progress.csv")
    
    group_cols = ["graph_idx", "hidden", "lr", "neg_ratio"]

    selected = df.groupby(group_cols).apply(
        lambda g: pd.Series({
            "Q_true": g["Q_true"].iloc[0],
            "Q_AIC": g.loc[g["AIC"].idxmax(), "Q_tested"],
            "Q_BIC": g.loc[g["BIC"].idxmax(), "Q_tested"],
            "Q_ICL": g.loc[g["ICL"].idxmax(), "Q_tested"],
        })
    ).reset_index()

    # --- correct flag ---
    selected["AIC_correct"] = (selected["Q_AIC"] == selected["Q_true"]).astype(int)
    selected["BIC_correct"] = (selected["Q_BIC"] == selected["Q_true"]).astype(int)
    selected["ICL_correct"] = (selected["Q_ICL"] == selected["Q_true"]).astype(int)
    print(pd.crosstab(selected["Q_true"], selected["Q_AIC"], rownames=["Q_true"], colnames=["Q_AIC"]))
    # Résumé
    summary = selected[["AIC_correct", "BIC_correct", "ICL_correct"]].mean().reset_index()
    summary.columns = ["critere", "success_rate"]
    print(summary)

    # --- plot propre ---
    plt.figure(figsize=(6,4))
    plt.bar(summary["critere"], summary["success_rate"], color="skyblue")
    plt.ylabel("Proportion correcte")
    plt.xlabel("Critère")
    plt.title("Critère sélectionnant le bon Q")
    plt.ylim(0,1)
    plt.grid(axis='y', alpha=0.3)
    plt.show()

def mainARi():
    df = pd.read_csv("results/hyperparams/benchmark_progress.csv")

    group_cols = ["graph_idx", "hidden", "lr", "neg_ratio"]

    # Sélectionner le Q avec ARI maximal
    selected_ARI = df.groupby(group_cols).apply(
        lambda g: pd.Series({
            "Q_true": g["Q_true"].iloc[0],
            "Q_ARI": g.loc[g["ARI"].idxmax(), "Q_tested"],
            "ARI_best": g["ARI"].max()
        })
    ).reset_index()

    # Score de réussite : Q_ARI == Q_true
    selected_ARI["ARI_correct"] = (selected_ARI["Q_ARI"] == selected_ARI["Q_true"]).astype(int)

    print("\n📌 Taux de réussite ARI pour la sélection du bon Q :")
    print(selected_ARI["ARI_correct"].mean())

    mean_val = selected_ARI["ARI_correct"].mean()

    plt.figure(figsize=(4,4))
    plt.bar(["ARI"], [mean_val])
    plt.ylim(0,1)
    plt.ylabel("Proportion correcte")
    plt.title("Taux de sélection du bon Q via ARI")
    plt.grid(axis="y")
    plt.show()

def mainHidden():
    df = pd.read_csv("results/hyperparams/benchmark_progress.csv")

    # --- Convertir 'hidden' de chaîne → vrai int ---
    df["hidden"] = df["hidden"].apply(lambda s: ast.literal_eval(s)[0])

    group_cols = ["graph_idx", "hidden", "lr", "neg_ratio"]

    # Sélection du Q pour chaque critère
    selected = df.groupby(group_cols).apply(
        lambda g: pd.Series({
            "Q_true": g["Q_true"].iloc[0],
            "Q_AIC": g.loc[g["AIC"].idxmax(), "Q_tested"],
            "Q_BIC": g.loc[g["BIC"].idxmax(), "Q_tested"],
            "Q_ICL": g.loc[g["ICL"].idxmax(), "Q_tested"],
        })
    ).reset_index()

    # Ajout colonnes de réussite
    selected["AIC_correct"] = (selected["Q_AIC"] == selected["Q_true"]).astype(int)
    selected["BIC_correct"] = (selected["Q_BIC"] == selected["Q_true"]).astype(int)
    selected["ICL_correct"] = (selected["Q_ICL"] == selected["Q_true"]).astype(int)

    # ===========================
    # 🔥 Analyse par nombre de couches
    # ===========================

    summary_by_hidden = selected.groupby("hidden")[["AIC_correct", "BIC_correct", "ICL_correct"]].mean()

    print("\n=== Performance par nombre de couches ===")
    print(summary_by_hidden)

    summary_by_hidden.plot(kind="bar", figsize=(8,5), title="Performance selon le nombre de couches")
    plt.ylabel("Taux de réussite")
    plt.ylim(0,1)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.show()

def mainARI_by_hidden():

    print("🔍 Analyse ARI en fonction du nombre de couches (hidden size)...")

    df = pd.read_csv("results/hyperparams/benchmark_progress.csv")

    # -------------------------------
    # 1) Parsing robuste de la colonne "hidden"
    # -------------------------------
    def parse_hidden(x):
        try:
            val = ast.literal_eval(x)
            if isinstance(val, list):
                return val[0]  # ex: "[32]" → 32
            return int(val)
        except:
            return int(x)  # fallback si c’est juste "32"

    df["hidden"] = df["hidden"].apply(parse_hidden)

    # -------------------------------
    # 2) Calcul ARI_correct : bon Q sélectionné ?
    # -------------------------------
    df["ARI_correct"] = (df["Q_selected"] == df["Q_true"]).astype(int)

    # -------------------------------
    # 3) Moyenne ARI_correct par hidden size
    # -------------------------------
    summary = df.groupby("hidden")["ARI_correct"].mean()
    print("\n📌 Taux de réussite ARI par taille de hidden :")
    print(summary)

    # -------------------------------
    # 4) Affichage graphique
    # -------------------------------
    plt.figure(figsize=(6, 4))
    summary.plot(kind="bar", color="steelblue")
    plt.title("ARI_correct en fonction du nombre de neurones hidden")
    plt.ylabel("Proportion de bons Q")
    plt.xlabel("Taille de hidden")
    plt.ylim(0, 1)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()

    print("\n🔥 Analyse ARI par hidden terminée.")


if __name__ == "__main__":
    print("Analyse....")
    mainARI_by_hidden()