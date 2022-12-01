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
genes = int(sys.argv[1])
network = nx.scale_free_graph(genes,0.4,0.3,0.3)
network = nx.to_pandas_edgelist(network)
network = network.values
network = np.hstack((network,np.random.normal(0,1,(network.shape[0],1))))
print(network)

# turn that in to an adjacency matrix 
adj = gs.util.numpy_to_adjacency(network)

# build a model by first starting an empty model 
#model = gs.GRNmodel()
#model.make_model(adj,comb)
#print(model.model)
#exec(model.model,globals())

# build a model from make_model 
M = gs.GRNmodel.make_model(adj)
M.genes = np.unique(network[:,0:2]).astype(int).tolist()

#print(M.parameters.start_rna,"|",M.parameters.time,"|",M.parameters.alphas,"|",M.parameters.dis_const)
#M.parameters.dis_const = 10
#print(M.parameters.start_rna,"|",M.parameters.time,"|",M.parameters.alphas,"|",M.parameters.dis_const)
#exec(M.model,globals())

M.parameters.generateMP()
M._start_model()
M.train()
M.get_pert("diagonal")
data = M.experiment()
print(data["time_100"])
