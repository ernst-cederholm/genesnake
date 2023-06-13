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

# make a random network 
#genes = int(sys.argv[1])
#network = nx.scale_free_graph(genes,0.4,0.3,0.3)
#network = nx.to_pandas_edgelist(network)
#network = network.values
#network = np.hstack((network,np.random.normal(0,1,(network.shape[0],1))))

network = gs.grn.make_random(5,2)
# turn that in to an adjacency matrix 
#adj = gs.util.numpy_to_adjacency(network)
# build a model from make_model 
M = gs.GRNmodel.make_model(network)
#print(network)
M.parameters.generateMP()
M.set_pert("diag",effect=(0.6,0.8),noise=0,multipert=3,exp=1,reps=2,sign=-1)
M.simulate_data(exp_type="ss",SNR=5,draw_pattern="lin",time_points=100,noise_model="gaussian")

print(M.model)
print(M.noise_free_data)
print(M.data)
#G = nx.from_pandas_adjacency(network,create_using=nx.DiGraph)
#nx.draw(G,with_labels=True)
sns.heatmap(M.noise_free_data,cmap="YlGn",linecolor="k",linewidths=0.01)
#plt.show()
#sns.heatmap(M.data,cmap="YlGn",linecolor="k",linewidths=0.01)
#plt.show()
#sns.heatmap(M.data-M.noise_free_data,cmap="YlOrRd",linecolor="k",linewidths=0.01)
#plt.show()

#M.save("test_GS_GRNmodel.pickle")

#loadmodel = gs.GRNmodel.load("test_GS_GRNmodel.pickle")
#for key in loadmodel.__dict__.keys():
#    if key == "parameters":
#        print(key,loadmodel.__dict__[key].__dict__)
#    else:
#        print(key,loadmodel.__dict__[key])
