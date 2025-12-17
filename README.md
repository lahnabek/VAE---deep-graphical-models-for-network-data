# Variational Auto-Encoders and Deep Graphical Models for Network Data


Implementation of DeepLPBM algorithm fro graph clustering, based on:

**"The Deep Latent Position Block Model For The Block Clustering And Latent Representation Of Networks"**

## Project Overview

This project was realized in the context of the Introduction to Probabilsitic graphcial Models and Deep Generative Models course taught by Pierre Latouche and Pierre-Alexandre Mattei in the MVA master's program. 

For a quick exposition of the main algorithm DeepLPBM, we invite the reader to go through **exposition.ipynb**. For a more thorough examination, below we detail the workings of all files in the project.


## Installation

## Requirements 


To run all files except autres_algos.py and evaluation.py:

numpy
pandas
matplotlib
sklearn
torch
scipy
torch_geometric
seaborn
networkx


To run autres_algos.py and evaluation.py:

graph_tool
igraph
leidenalg
graspologic
 

## Dataset



## Experiments



## Code Structure

### Python Files



- **class_GCNEncoder**: Contains the GCN Encoder architecture for the main algorithm.

- **deep_lpbm.py**: Implementation of the Deep LPBM algorithm. It depends on "class_GCNEncoder.py". Can be run from the command line, using its internal config, or its main function "main" can be imported.

- **synthetic_data.py**: Serves for creation of synthetic data according to partial membership stochastic block model. Can be run from command line, or its main function "generate_synthetic" can be imported. It produces matrices $A$, hard labels $y$, soft labels $eta$, connectivity matrices $\Pi$ as .npy files in the directory data_synthetic. 

- **autres_algos.py**: Contains wrappers for algorithms different form DeepLPBM, requires all additional libraries to run. 

- **evaluation.py**: Comparison algorithm. It depends on "deep_lpbm.py", "synthetic_data-py", "autres_algos.py". In particular, it requires all additional libraries. It is run from the command line.

- **data.py**: Used to turn HCP-100 into usable connectivity graphs. 

- **hyperparameters, 1, 2, 3, 4.py**: hyperparameters test on various types of sythetic data.

- **hpbench.py, hpaalysis.py**: analyse des hyperparametres. 


### Notebooks

- **human:brains_exp.ipynb**: Experiment with HCP-100 dataset. 

- **exposition.ipynb**: Expository file, contains the most important functions from "deep_lpbm.py" and some comments. 


### Other directories
- data_synthetic: Directory where synthetic data is stored. Anytime synthetic_data.py is run, this gets rewritten.
- dataset_numpy_spanningtree: The processed HCP-100 dataset.
- human_brains_networks_results: Where we store results on the experiment on the HCP-100 dataset.
- Comparison: Directory where we store results for comparison experiments.
- results: general directory where deep_lpbm stores results unless differently propmted.




## Paper Reference

```

@article{DeepLPBM,
title = "The Deep Latent Position Block Model for Block Clustering and Latent Representation of Nodes in Networks",
keywords = "Node partial memberships, block modelling, graph variational autoencoder, graph visualisation, positional modelling",
author = "R{\'e}mi Boutin and Pierre Latouche and Charles Bouveyron",
note = "Publisher Copyright: {\textcopyright} The Author(s), under exclusive licence to Springer Science+Business Media, LLC, part of Springer Nature 2025.",
year = "2025",
month = oct,
day = "1",
doi = "10.1007/s11222-025-10679-7",
language = "English",
volume = "35",
journal = "Statistics and Computing",
issn = "0960-3174",
number = "5",

}
'''