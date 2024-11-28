# MG
# load libraries

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from methods.grn_inference import infer_networks, cutoff_handler

# sample Y matrix (sY) 4x4 for 1 replicate
# 1st replicate
sY1 = pd.DataFrame({"G1": [-1, 0, -0.65, 0],
                    "G2": [-0.2, -0.98, 0, 0.01],
                    "G3": [-0.1, 0.09, -1, 0.2],
                    "G4": [0, -0.09, 0.1, -0.96]})
# 2nd replicate
sY2 = pd.DataFrame({"G1_2": [-0.9, 0.1, -0.7, 0],
                    "G2_2": [-0.05, -1, 0, 0.4],
                    "G3_2": [0, -0.04, -0.88, 0.6],
                    "G4_2": [0, 0.02, 0.3, -0.99]})
# merge replicates
sY = pd.concat([sY1, sY2], axis=1)

# sample P matrix (sP) 4x8
sP1 = np.eye(4, 4)  # create matrix with 1's on diagonal (for 1 replicate)
sP1 = pd.DataFrame(sP1)
sP = -pd.concat([sP1, sP1], axis=1)  # as two replicates


# make N networks under various cutoffs using vector of zeta values as thresholds
# auto means that it takes 100 thresholds between 0 and maximum absolute value from the full network



# function for inferring networks



# testing the GRN inference
zvec = np.logspace(-6, 0, 30)
Anets = infer_networks(sY, sP, "rbfsvm", zvec)

# simple visualization of network
# select n-th network and display it
n = 21
print(Anets[n])
net_to_show = pd.DataFrame(Anets[n])

G = nx.to_networkx_graph(net_to_show.transpose(), create_using=nx.DiGraph()) # directed network
G.remove_edges_from(nx.selfloop_edges(G))  # remove selfloops
pos = nx.circular_layout(G) # set up layout
wgs = nx.get_edge_attributes(G, "weight") # give it a weight
# draw the network with visible labels, opacity 50% and arrows
nx.draw(G, pos, with_labels=True, node_size=1000, alpha=0.5, font_weight="bold", arrows=True, arrowstyle='simple', node_color='gray')
plt.axis('on')
plt.show()
