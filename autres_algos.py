'''
Contient tous les algorithmes de clustering utiles pour faire l'evaluation de l'algorithme.
Importante: il utilizes des bibliotèques différents du reste des fichiers
'''

import numpy as np

import igraph as ig
import leidenalg
import networkx as nx
import graph_tool.all as gt

from sklearn.cluster import SpectralClustering

from sklearn.manifold import spectral_embedding
from sklearn.cluster import KMeans



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



