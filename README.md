# Variational Auto-Encoders and Deep Graphical Models for Network Data


Implementation of DeepLPBM algorithm fro graph clustering, based on:

**"The Deep Latent Position Block Model For The Block Clustering And Latent Representation Of Networks"**

## Project Overview

This project was realized in the context of the Introduction to Probabilsitic graphcial Models adn Deep Generative Models course taught by Pierre Latouche and Pierre-Alexandre Mattei in the MVA master's program. 

It implements a two experiments challenging the limitations of the afored mention paper. The code is made to be reproducible and well-documented.


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


To run evaluation.py:

igraph
leidenalg
graph_tool
graspologic
community
 

## Dataset

## Experiments



## Code Structure

### Python Files

- **class_GCNEncoder**: Contains the GCN Encoder architecture for the main algorithm.

- **deep_lpbm.py**: Implementation of the Deep LPBM algorithm. It depends on "class_GCNEncoder.py". Can be run from the command line, using its internal config, or its main function "main" can be imported.

- **synthetic_data.py**: Serves for creation of synthetic data according to partial membership stochastic block model. Can be run from command line, or its main function "generate_synthetic" can be imported. 

- **autres_algos.py**: Contains wrappers for algorithms different form DeepLPBM, requires **graph_tool** to run. 

- **evaluation.py**: Comparison algorithm. It depends on "deep_lpbm.py", "synthetic_data-py", "autres_algos.py". In particular, it requires **graph_tool**. It is run from the command line.


### Notebooks

- **human:brains_exp.ipynb**: Experiment with HCP-100 dataset. 
- **exposition.opynb**: Expository file, contains the most important functions from "deep_lpbm.py" and some comments. 




## Paper Reference

```

@article{DeepLPBM,
title = "The Deep Latent Position Block Model for Block Clustering and Latent Representation of Nodes in Networks",
abstract = "The current surge in data has led to a significant increase in the size of networks used to model relationships between different objects represented as nodes. Therefore, summarising network information is a crucial task that can be conducted using node clustering methods. Additionally, to ensure interpretable results, it is essential to employ relevant visualisation techniques to depict the network. To tackle both issues, we propose a new methodology called the deep latent position block model (Deep LPBM). This simultaneously provides a network visualisation coherent with block modelling, allowing a clustering more general than community detection methods, as well as a continuous representation of nodes in a latent space given by partial membership vectors. Deep LPBM is based on a variational autoencoder strategy, relying on a graph convolutional network, with a specifically designed decoder. The inference involves the construction of an approximation of the marginal likelihood of Deep LPBM through the expected lower bound (ELBO). A gradient-descent algorithm based on Monte-Carlo approximations is used to optimize the ELBO with respect to its parameters. To select the number of clusters, we compare three model selection criteria. A node clustering benchmark comprising positional community detection as well as model methods is conducted. We also compare the quality of Deep LPBM node partial membership estimation with other methodologies. We conclude with an analysis of the French political blogosphere network and a comparison with another methodology to illustrate the novelty provided by Deep LPBM results.",
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