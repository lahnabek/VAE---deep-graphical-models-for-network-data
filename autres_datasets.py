#   AUTRES DATA SETS 

import graph_tool.all as gt
from graph_tool.spectral import adjacency 
from graph_tool import collection
import numpy as np
import os
import networkx as nx
from scipy.io import mmread
import matplotlib.pyplot as plt

collections = ['karate', 'dolphins', 'polbooks']

save_dir = os.path.join(os.getcwd(), "miscdata") 


for collec in collections: 
    g = collection.data[collec]
    A = adjacency(g)
    A = A.toarray()
    save_path = os.path.join(save_dir, "A_"+ collec +".npy")
    np.save(save_path, A)

print(f"Adjacency matrix saved to: {save_path}")




lesmis = os.path.join(os.getcwd(),"lesmis/lesmis.mtx")

A = mmread(lesmis) 
A = A.toarray()

save_path = os.path.join(save_dir, "A_lesmis.npy")
np.save(save_path, A)




edges = []
with open("ia-southernwomen.edges") as f:
    for line in f:
        if line.startswith("%") or not line.strip():
            continue
        w, e = line.strip().split()  # two columns: woman, event
        # prepend letters to distinguish node sets
        edges.append(("w" + w, "e" + e))  # w = woman, e = event

# Create bipartite graph
B = nx.Graph()
B.add_edges_from(edges)

plt.figure(figsize=(6,6))
nx.draw(B, with_labels=True, node_size=300)
plt.show()

print("Nodes:", B.number_of_nodes())
print("Edges:", B.number_of_edges())

nodelist = sorted(B.nodes()) 
A = nx.to_numpy_array(B, nodelist=nodelist)
print(A.shape)

save_path = os.path.join(save_dir, "A_southernwomen.npy")
np.save(save_path, A)