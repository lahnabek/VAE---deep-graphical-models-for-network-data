#Contient tous les algorithmes de clustering utilisables


import os
import glob
import json

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
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score as ARI
from graspologic.models import SBMEstimator
from graspologic.embed.ase import AdjacencySpectralEmbed
from sklearn.mixture import GaussianMixture
from sklearn.manifold import spectral_embedding
from sklearn.cluster import KMeans



#---------------------
#Choses pour R
#----------------------

import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector

def install_r_dependencies():
    """Installs required R packages for VBLPCM."""
    # Import R's utility package
    utils = rpackages.importr('utils')
    
    # Select the first CRAN mirror to avoid interactive popups
    utils.chooseCRANmirror(ind=1)
    
    # Packages to install
    needed_packs = ['vblpcm', 'network', 'sna']
    
    # Check which are missing
    missing_packs = [x for x in needed_packs if not rpackages.isinstalled(x)]
    
    if missing_packs:
        print(f"Installing missing R packages: {missing_packs}...")
        utils.install_packages(StrVector(missing_packs))
    else:
        print("All R packages are already installed.")

# Run the installation
install_r_dependencies()


#--------------------------------------------
#OUTIL
#--------------------------------------------

def compute_block_density(A, z, K):
    """
    Computes B = Z.T @ A @ Z normalized by block sizes.
    Much faster than for-loops.
    """
    # Create One-Hot encoding of z
    N = A.shape[0]
    Z_mat = np.zeros((N, K))
    for i, cluster in enumerate(z):
        if 0 <= cluster < K:
            Z_mat[i, int(cluster)] = 1

    # Count nodes in each cluster
    n_counts = Z_mat.sum(axis=0) # shape (K,)

    # Sum edges between clusters: M = Z.T * A * Z
    # M[r,s] = sum of edges between block r and s
    M = Z_mat.T @ A @ Z_mat 

    B = np.zeros((K, K))
    
    for r in range(K):
        for s in range(K):
            if n_counts[r] == 0 or n_counts[s] == 0:
                continue
            
            if r == s:
                # Undirected: max edges is nr*(nr-1) if we exclude self-loops
                possible = n_counts[r] * (n_counts[r] - 1)
                # A is symmetric, so M[r,r] counts edges twice. 
                # If A is binary: M[r,r] is 2 * num_edges.
                # Density = M[r,r] / possible. 
                # Note: if A has self-loops, math changes slightly. Assuming no self-loops.
                if possible > 0:
                    B[r, s] = M[r, s] / possible
            else:
                possible = n_counts[r] * n_counts[s]
                if possible > 0:
                    B[r, s] = M[r, s] / possible
                    
    return B



#--------------------------------------------------------------
#Algorithmes
#-----------------------------------------------------------------


#SBM avec graph_tool

def run_graphtool_sbm(A, K=None):
    # 1. Convert Numpy Adjacency to Graph-Tool Graph
    g = gt.Graph(directed=False)
    sources, targets = A.nonzero()
    edge_list = np.column_stack((sources, targets))
    
    # Filter for undirected (keep only source <= target to avoid duplicates)
    if not g.is_directed():
        mask = edge_list[:, 0] <= edge_list[:, 1]
        edge_list = edge_list[mask]
        
    g.add_edge_list(edge_list)

    # 2. Run SBM Inference
    state = gt.minimize_blockmodel_dl(g)

    # --- FIX STARTS HERE ---
    
    # 3. Extract and Remap Block Assignments
    # z_raw contains the original labels (e.g., 0, 5, 10)
    z_raw = state.get_blocks().get_array()
    
    # np.unique with return_inverse=True does two things:
    # 'unique_labels': The sorted original labels (e.g., [0, 5, 10])
    # 'z': The remapped labels 0..K-1 (e.g., [0, 1, 2]) matching the original structure
    unique_labels, z = np.unique(z_raw, return_inverse=True)
    K = len(unique_labels)
    
    # Calculate size of each block (n_r) using the remapped z
    # Since z is now 0..K-1, counts corresponds exactly to indices 0..K-1
    _, n_r = np.unique(z, return_counts=True)
    
    # 4. Compute Connectivity Matrix (B)
    # The state matrix uses the ORIGINAL labels as indices.
    # It might be a large sparse matrix if labels are high numbers (e.g. 10).
    m_counts = state.get_matrix()
    E_full = np.array(m_counts.todense())
    
    # We extract only the sub-matrix corresponding to our existing blocks.
    # np.ix_ allows us to grab rows [0, 5, 10] and columns [0, 5, 10]
    E_rs = E_full[np.ix_(unique_labels, unique_labels)]
    
    # Initialize Probability Matrix B
    # Compute B fast
    B = compute_block_density(A, z, K)

    return {"z": z, "K": K, "B": B}


#Spectral clustering avec sklearn 


def run_spectral_clustering(A, K=3):
    sc = SpectralClustering(
        n_clusters=K,
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=42
    )
    z = sc.fit_predict(A)
    # Compute B for consistency (optional)
    B = compute_block_density(A, z, K)
    return {"z": z, "K": K, "B": B}


#soft spectral clustering avec sklearn


def run_soft_spectral_clustering(A, K=3):
    """
    Spectral Clustering that returns soft assignments (probabilities).
    """
    # FIX: Ensure matrix is float for spectral calculations
    A_float = A.astype(float)

    # 1. Spectral Embedding
    maps = spectral_embedding(A_float, n_components=K, drop_first=False)
    
    # ... rest of the function remains the same ...
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10).fit(maps)
    z = kmeans.labels_
    dists = kmeans.transform(maps)
    
    alpha = 1.0 
    exp_dists = np.exp(-alpha * dists)
    sum_exp = exp_dists.sum(axis=1, keepdims=True)
    eta = exp_dists / np.maximum(sum_exp, 1e-10)
    
    return {"z": z, "K": K, "eta": eta}
#louvain


def run_louvain(A, K=None):
    G = nx.from_numpy_array(A)
    # nx.community.louvain_communities returns list of sets
    communities = nx.community.louvain_communities(G, seed=42)
    
    K = len(communities)
    z = np.zeros(A.shape[0], dtype=int)
    for k, nodes in enumerate(communities):
        for node in nodes:
            z[node] = k
            
    return {"z": z, "K": K}



#leiden

def run_leiden(A, K=None):
    g = ig.Graph.Adjacency((A > 0).tolist(), mode="undirected")
    partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, seed=42)
    z = np.array(partition.membership)
    K = len(np.unique(z))
    
    B = compute_block_density(A, z, K)
    return {"z": z, "K": K, "B": B}




#VBLPCM avec Python


def run_vblpcm_python(A, max_k=10, K=None):
    # 1. Embed (ASE)
    ase = AdjacencySpectralEmbed(n_components=None) 
    X = ase.fit_transform(A)
    if isinstance(X, tuple):
        X = np.concatenate(X, axis=1)

    # 2. GMM with BIC selection
    best_gmm = None
    lowest_bic = np.inf 
    
    # Limit max_k to N/2 to avoid errors on small graphs
    limit_k = min(max_k, A.shape[0] // 2)
    
    for k in range(1, limit_k + 1):
        try:
            gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
            gmm.fit(X)
            bic = gmm.bic(X)
            if bic < lowest_bic:
                lowest_bic = bic
                best_gmm = gmm
        except Exception:
            continue
            
    if best_gmm is None:
        z = np.zeros(A.shape[0], dtype=int)
        eta = np.ones((A.shape[0], 1)) # Trivial soft assignment
        K = 1
    else:
        z = best_gmm.predict(X)
        K = best_gmm.n_components
        eta = best_gmm.predict_proba(X)
    
    return {"z": z, "K": K, "eta": eta}



#VBLPCM avec R, faire gaffe peut etre qu'il est très lent


def run_vblpcm_r(A, max_k=5, K=None):
    """
    Robust R implementation. 
    Assumes install_r_dependencies() has been run.
    """
    is_directed = not np.allclose(A, A.T)
    r_directed = "TRUE" if is_directed else "FALSE"

    # Activate numpy converter
    with localconverter(default_converter + numpy2ri.converter):
        ro.globalenv["A"] = A
        
        # Robust R script
        r_script = f"""
        library(vblpcm)
        library(network)
        
        # Suppress warnings for cleaner output
        suppressWarnings({{
            net <- network(as.matrix(A), directed={r_directed})
            
            best_bic <- -Inf
            best_z <- rep(1, network.size(net)) # Default fallback
            
            # Search G from 1 to {max_k}
            for (g in 1:{max_k}) {{
                tryCatch({{
                    v_start <- vblpcmstart(net, G=g, plot=FALSE, verbosity=0)
                    v_fit <- vblpcmfit(v_start, STEPS=20, plotting=FALSE) # Lower STEPS for speed
                    
                    if (!is.nan(v_fit$BIC) && v_fit$BIC > best_bic) {{
                        best_bic <- v_fit$BIC
                        # Extract hard assignments from probabilities
                        best_z <- apply(v_fit$Y, 1, which.max)
                    }}
                }}, error=function(e){{ NULL }})
            }}
        }})
        best_z
        """
        
        try:
            # Execute R code and get result
            z_r = ro.r(r_script)
            z_raw = np.array(z_r, dtype=int)
            z = z_raw - 1 # R is 1-indexed
        except Exception as e:
            print(f"R VBLPCM Error: {e}")
            # Fallback
            z = np.zeros(A.shape[0], dtype=int)

    K = len(np.unique(z))
    B = compute_block_density(A, z, K)
    
    return {"z": z, "K": K, "B": B}





