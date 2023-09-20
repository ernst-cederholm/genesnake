from scipy.integrate import odeint
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import networkx as nx 
import pandas as pd
import numpy as np
import itertools
import random 
import sys

import genesnake as gs
from methods.grn_inference import infer_networks

# make a random network 
#genes = int(sys.argv[1])
#network = nx.scale_free_graph(genes,0.4,0.3,0.3)
#network = nx.to_pandas_edgelist(network)
#network = network.values
#network = np.hstack((network,np.random.normal(0,1,(network.shape[0],1))))

for i in range(1000):
    genes = 100 #random.randint(10,1000)
    s = round(random.uniform(2,10),2)
    print("\ntest #"+str(i)+" number of genes:"+str(genes)+" sparcity:"+str(s))
    network = gs.grn.make_FFLATnetwork(NETWORK_SIZE=genes,SPARSITY=s)
    
    #G = nx.from_pandas_adjacency(network,create_using=nx.DiGraph)
    #nx.draw_spring(G,with_labels=True)
    #plt.show()
    
    #print(network)
    
    # build a model from make_model 
    #M = gs.GRNmodel.make_model(network)
    #print(M.model)
