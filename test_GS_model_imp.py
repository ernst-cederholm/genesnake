from scipy.integrate import odeint
import matplotlib.pyplot as plt
import scipy.stats as stats
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

network = gs.grn.make_random(int(sys.argv[1]),3) #make_FFLATnetwork(network_size=sys.argv[1])
# turn that in to an adjacency matrix 
#adj = gs.util.numpy_to_adjacency(network)
# build a model from make_model 
M = gs.ODEmodel.make_ODEmodel(network)

M.parameters.generateMP()
M.train()

M.set_pert("diag",effect=(0.6,0.8),noise=0,multipert=3,reps=2,sign=1)
data = M.experiment()
print(data["time_100"])
#M.save("test_GS_GRNmodel.pickle")

#loadmodel = gs.GRNmodel.load("test_GS_GRNmodel.pickle")
#for key in loadmodel.__dict__.keys():
#    if key == "parameters":
#        print(key,loadmodel.__dict__[key].__dict__)
#    else:
#        print(key,loadmodel.__dict__[key])
