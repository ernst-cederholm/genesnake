from scipy.integrate import odeint
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import networkx as nx
import pandas as pd
import numpy as np
import itertools
import sys

import genesnake as gs

genes = 100
network = gs.grn.make_FFLATnetwork.make_FFLATTnetwork(NETWORK_SIZE=genes,FFL_ENRICHED=0,self_loops=False)

print(network)
print("activating edges:",(network.values>0).sum())
print("inhibitory edges:",(network.values<0).sum())

G = nx.from_pandas_adjacency(network,create_using=nx.DiGraph())
edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
colors = list()
for w in weights:
    if w < 0:
        colors.append("r")
    else:
        colors.append("b")
nx.draw_spring(G,with_labels=True,edgelist=edges,edge_color=colors)
plt.show()



M = gs.GRNmodel.make_model(network)
M.set_pert("diag",reps=3)
M.simulate_data(exp_type="steady-state",SNR=0.5,noise_model="normal")
#print(M.data)
#print(M.noise_free_data)
#gs.analysis.snakeda(M.noise_free_data,M.perturbation)
gs.analysis.snakeda(M.noise_free_data,M.perturbation,out="GS_EDA.html")
