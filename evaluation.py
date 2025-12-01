import os
import glob
import json

import rpy2.robjects as ro
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import default_converter

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

import igraph as ig
import leidenalg
import networkx as nx
import graph_tool.all as gt
from graph_tool.all import Graph, minimize_blockmodel_dl
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.cluster import SpectralClustering
from graspologic.models import SBMEstimator
import community as community_louvain
from sklearn.mixture import GaussianMixture
from graspologic.embed.ase import AdjacencySpectralEmbed

import autres_algos as AA

from deep_lpbm_copy import main as run_deepLPBM
from deep_lpbm_copy import draw_graph_hard_clusters
from synthetic_data import generate_synthetic

#-------------------------------------
# CONFIG
#-------------------------------------
DATA_DIR = "data_numpy_synthetic"  #"dataset_numpy_spanningtree" #"miscdata" 
ALL = False             # Évaluer tous les sujets dans DATA_DIR
SUBJECT_IDX = 9         # Sujet à évaluer DANS DATA_DIR
K = 3                    # Nombre de clusters (si applicable)
results_dir = "Comparison"



# ------------------------------------
# Evaluation functions
#-----------------------------------------


def compute_H(eta, eta_prime, N):
    U = eta @ eta.T
    U_prime = eta_prime @ eta_prime.T
    K = eta.shape[0]
    
    # Correct logic for symmetric matrix difference norm
    diff = np.abs(U - U_prime)
    # Sum lower triangle (excluding diagonal) * 2 + diagonal
    l1_difference = np.sum(np.tril(diff, k=-1)) * 2 + np.sum(np.diag(diff))

    return np.sqrt(2*l1_difference/(N*(N-1)))


def hard_to_soft(z, K):
    """Converts labels [0, 1, 0...] to One-Hot Matrix."""
    N = len(z)
    eta = np.zeros((N, K))
    for i, val in enumerate(z):
        if 0 <= val < K:
            eta[i, int(val)] = 1.0
    return eta

def evaluate_algorithm(A: np.array, result: dict, z_true=None, eta_true=None): 
    z = result["z"]
    K = result["K"]

    metrics = {}

    # 1) ARI
    if z_true is not None:
        from sklearn.metrics import adjusted_rand_score
        # Ensure z is integer type
        metrics["ARI"] = adjusted_rand_score(z_true, z.astype(int))

    # 2) PME (Partial Membership Error / H score)
    if eta_true is not None:
        # If result only has hard z, convert to one-hot eta
        if "eta" in result and result["eta"] is not None:
            eta_est = result["eta"]
        else:
            eta_est = hard_to_soft(z, K)
            
        metrics["H"] = compute_H(eta_est, eta_true, z.shape[0])


    # 3) Normalised mutual information score 
    if z_true is not None:
        from sklearn.metrics import normalized_mutual_info_score
        metrics["NMI"] = normalized_mutual_info_score(z_true, z.astype(int))


    metrics["K_est"] = K
    metrics["z"] = z
    return metrics



# ---------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------




def list_npy_graphs(data_dir: str):
    """
    Retourne les chemins des fichiers .npy dont le nom commence par 'A'.
    """
    return sorted(glob.glob(os.path.join(data_dir, "A*.npy")))



def load_A(path: str, as_bool: bool = True, validate: bool = True) -> np.array:
    """
    Charge une matrice A (carrée, symétrique, diag=0 si validate).
    as_bool: binarise A (uint8) si True.
    """
    A = np.load(path)
    if validate:
        assert A.ndim == 2 and A.shape[0] == A.shape[1], "A doit être carrée."
        if np.issubdtype(A.dtype, np.floating):
            assert np.allclose(A, A.T, atol=1e-8), "A doit être symétrique."
        else:
            assert (A == A.T).all(), "A doit être symétrique."
        assert np.allclose(np.diag(A), 0), "diag(A) doit être nulle."
    if as_bool:
        A = (A != 0).astype(np.uint8)
    return A



def run_experiment(A: np.array, z_true: list = None, eta_true: np.array = None, algorithms: dict = None):
    """
    Runs a suite of algorithms on the graph A.
    """
    results = {}

    for name, algo in algorithms.items():
        print(f"Running {name}...")
        # Run the specific algorithm
        if eta_true is not None: 
            result = algo(A, K=eta_true.shape[1])
        else: result = algo
        
        # Evaluate the result immediately
        # Note: We pass eta_true here so H-score can be calculated
        metrics = evaluate_algorithm(A, result, z_true, eta_true)
        results[name] = metrics

    return results



def main(DATA_DIR=DATA_DIR, SUBJECT_IDX=SUBJECT_IDX, results_dir= results_dir ):
    # --- 1. Load Files ---
    files = list_npy_graphs(DATA_DIR)
    assert len(files) > 0, f"Aucun fichier .npy trouvé dans {DATA_DIR}"
    # print(f"{len(files)} matrices détectées dans {DATA_DIR}")

    # Determine which files to process
    if not ALL:
        files_to_process = [files[SUBJECT_IDX]]
        indices_to_process = [SUBJECT_IDX]
    else:
        files_to_process = files
        indices_to_process = range(len(files))

    # Prepare directories
    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)
        ari_path = os.path.join(results_dir, "ARI.txt")
        pme_path = os.path.join(results_dir, "partial_membership_evaluation.txt")
        nmi_path = os.path.join(results_dir, "NMI.txt")
        img_path = os.path.join(results_dir, "images")
        os.makedirs(img_path, exist_ok=True)

        if not os.path.exists(ari_path):
            with open(ari_path, 'w') as f: f.write("Graph,Algorithm,ARI\n")
        if not os.path.exists(pme_path):
            with open(pme_path, 'w') as f: f.write("Graph,Algorithm,H_Score\n")
        if not os.path.exists(nmi_path):
            with open(ari_path, 'w') as f: f.write("Graph,Algorithm,NMI\n")

    # Variables to store the results of the LAST graph processed
    # (This ensures compatibility with your zeta loop)
    final_ARI = {}
    final_PME = {}
    final_NMI = {}

    # --- 3. Main Loop ---
    for real_idx, A_path in zip(indices_to_process, files_to_process):
        # print(f"\n=== Processing Subject {real_idx} : {os.path.basename(A_path)} ===")

        # A. Load Data
        A = load_A(A_path)
        base_name = os.path.basename(A_path)
        
        y_true = None
        eta_true = None 
        
        y_path = os.path.join(DATA_DIR, base_name.replace("A_", "y_"))
        eta_path = os.path.join(DATA_DIR, base_name.replace("A_", "eta_"))

        if os.path.exists(y_path):
            y_true = np.load(y_path).astype(int).ravel()
        if os.path.exists(eta_path):
            eta_true = np.load(eta_path)


        # B. Run Standard Algorithms
        results = run_experiment(A, z_true=y_true, eta_true=eta_true, algorithms=algos)

        # C. Run Deep LPBM
        # print(f"Running Deep LPBM on index {real_idx}...")
        base_name = os.path.basename(A_path)
        current_id = int(base_name.split('_')[1].split('.')[0])

        deeplpbm_res = run_deepLPBM(DATA_DIR=DATA_DIR, SUBJECT_IDX=real_idx, comparison=True, Q_true=eta_true.shape[1])
        results['deepLPBM'] = evaluate_algorithm(A, deeplpbm_res, y_true, eta_true)

        # D. Save Results & Update Return Variables
        # Create a dictionary for this specific graph's scores
        current_ARI = {}
        current_PME = {}
        current_NMI = {}
        
        for algo, metrics in results.items():
            val_ari = metrics.get("ARI", None)
            val_h = metrics.get("H", None)
            val_nmi = metrics.get('NMI', None)
            
            current_ARI[algo] = val_ari
            current_PME[algo] = val_h
            current_NMI[algo] = val_nmi

        # Update the final return variables to match the current graph
        final_ARI = current_ARI
        final_PME = current_PME
        final_NMI = current_NMI

        if results_dir is not None:
            # Write to disk
            with open(ari_path, 'a') as f:
                for algo, score in current_ARI.items():
                    f.write(f"{base_name},{algo},{score}\n")

            with open(pme_path, 'a') as f:
                for algo, score in current_PME.items():
                    f.write(f"{base_name},{algo},{score}\n")

            with open(nmi_path, 'a') as f:
                for algo, score in current_NMI.items():
                    f.write(f"{base_name},{algo},{score}\n")

            # Save Images
            for algo_name, metrics in results.items():
                if "z" in metrics:
                    safe_name = f"{base_name.replace('.npy','')}_{algo_name}"
                    draw_graph_hard_clusters(A, metrics["z"], results_dir=img_path, algo=safe_name)

    # print("Processing complete.")
    
    return final_ARI, final_PME, final_NMI


#--------------------------------------------------------------------
#SETUP EXPERIMENT
#--------------------------------------------------------------------


algos = {
    "SBM_Python": AA.run_graphtool_sbm,
    "Spectral_Clustering": AA.run_spectral_clustering,
    "Soft_Spectral_Clustering": AA.run_soft_spectral_clustering,
    "Louvain": AA.run_louvain,
    "Leiden": AA.run_leiden,
    "fake_VBLPCM_python": AA.run_vblpcm_python 
    #"vblpcm_r": AA.run_vblpcm_r
}
#deep_LPBM est traité separement




mode = "assortative"   #"disassortative", "hub"

N = 150 #nombre de nodes
K = 4             #nombre de clusters
beta = 0.8                   #grande probabiité de connection
epsilon = 0.1                #petite probabilité de connextion
zetas = [j/10 for j in range(6, 11) ]      #niveau de bruit


outdir = "data_synthetic"
DATA_DIR =  outdir + "/" + mode
ALL = False             # Évaluer tous les sujets dans DATA_DIR
SUBJECT_IDX = 0          # Sujet à évaluer DANS DATA_DIR
results_dir = "Comparison"




def comparison_experiment():


    total_ARI = {algo: [] for algo in algos.keys()}
    total_ARI['deepLPBM'] = []
    
    total_PME = {algo: [] for algo in algos.keys()}
    total_PME['deepLPBM'] = []

    total_NMI = {algo: [] for algo in algos.keys()}
    total_NMI['deepLPBM'] = []


    for x in range(len(zetas)):
        print(f"\n--- Running Experiment for Zeta = {zetas[x]} ---")
        generate_synthetic(mode=mode, #disassortative, hub, disassortative
                       outdir=outdir,
                       n_graphs=1,
                       N=N,
                       Q=K,
                       beta=beta,
                       eps=epsilon,
                       zeta=zetas[x], # 1 for hard clustering and in (0,1) for partial
                       seed=0)
    
        partial_ARI, partial_PME, partial_NMI = main(DATA_DIR=DATA_DIR, 
                                        SUBJECT_IDX=SUBJECT_IDX,
                                        results_dir=None)
        print('ari', partial_ARI, 'pme', partial_PME, 'nmi', partial_NMI)

        for algo in partial_ARI.keys():
            print('algo update', algo)
            total_ARI[algo].append(partial_ARI[algo])
            total_PME[algo].append(partial_PME[algo])
            total_NMI[algo].append(partial_NMI[algo])
        print('total_ari', total_ARI)



    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)
        ari_path = os.path.join(results_dir, "total_ARI.txt")
        pme_path = os.path.join(results_dir, "total_partial_membership_evaluation.txt")
        nmi_path = os.path.join(results_dir, "total_NMI.txt")
        img_path = os.path.join(results_dir, "images")
        os.makedirs(img_path, exist_ok=True)

        with open(ari_path, 'w') as f: 
                # Header: Zeta, Algo1, Algo2, ...
                header = "Zeta," + ",".join(total_ARI.keys()) + "\n"
                f.write(header)
            
                # Rows
                print('ari', total_ARI)
                for i, z_val in enumerate(zetas):
                    print('i', i, 'z', z_val)
                    row_vals = [f"{total_ARI[algo][i]:.4f}" for algo in total_ARI.keys()]
                    f.write(f"{z_val}," + ",".join(row_vals) + "\n")

        with open(pme_path, 'w') as f:
            header = "Zeta," + ",".join(total_PME.keys()) + "\n"
            f.write(header)
            
            for i, z_val in enumerate(zetas):
                row_vals = [f"{total_PME[algo][i]:.4f}" for algo in total_PME.keys()]
                f.write(f"{z_val}," + ",".join(row_vals) + "\n")

        with open(nmi_path, 'w') as f: 
                # Header: Zeta, Algo1, Algo2, ...
                header = "Zeta," + ",".join(total_NMI.keys()) + "\n"
                f.write(header)
            
                # Rows
                for i, z_val in enumerate(zetas):
                    row_vals = [f"{total_NMI[algo][i]:.4f}" for algo in total_NMI.keys()]
                    f.write(f"{z_val}," + ",".join(row_vals) + "\n")

    #ARI Plot
    plt.figure(figsize=(10, 6))
    plt.xlabel("Zeta (Hardness of Clustering)")
    plt.ylabel("ARI (Adjusted Rand Index)")
    plt.title("Clustering Performance vs. Mixing Parameter")
    
    for algo, scores in total_ARI.items():
        plt.plot(zetas, scores, marker='o', label=algo)
        
    plt.legend()
    plt.grid(True, alpha=0.3)
    if results_dir:
        plt.savefig(os.path.join(results_dir, "ARI_comparison.png"), dpi=300)
    plt.show()
    plt.close() 


    # PME Plot
    plt.figure(figsize=(10, 6))
    plt.xlabel("Zeta (Hardness of Clustering)")
    plt.ylabel("PME (H-Score Error)")
    plt.title("Partial Membership Estimation Error")
    
    for algo, scores in total_PME.items():
        plt.plot(zetas, scores, marker='s', linestyle='--', label=algo)
        
    plt.legend()
    plt.grid(True, alpha=0.3)
    if results_dir:
        plt.savefig(os.path.join(results_dir, "PME_comparison.png"), dpi=300)
    plt.show()
    plt.close()


    #ARI Plot
    plt.figure(figsize=(10, 6))
    plt.xlabel("Zeta (Hardness of Clustering)")
    plt.ylabel("NMI (Normalised Mutual Information)")
    plt.title("Clustering Performance vs. Mixing Parameter")
    
    for algo, scores in total_ARI.items():
        plt.plot(zetas, scores, marker='o', label=algo)
        
    plt.legend()
    plt.grid(True, alpha=0.3)
    if results_dir:
        plt.savefig(os.path.join(results_dir, "NMI_comparison.png"), dpi=300)
    plt.show()
    plt.close() 




if __name__ == "__main__":
    comparison_experiment()



