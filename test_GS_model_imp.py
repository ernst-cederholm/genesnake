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
from methods.grn_inference import infer_networks

# make a random network 
genes = int(sys.argv[1])
#network = nx.scale_free_graph(genes,0.4,0.3,0.3)
#network = nx.to_pandas_edgelist(network)
#network = network.values
#network = np.hstack((network,np.random.normal(0,1,(network.shape[0],1))))

network = gs.grn.make_FFLATnetwork(NETWORK_SIZE=genes,FFL_ENRICHED=0)
#network = gs.grn.make_DAG(genes,1.2)
print(network)
# turn that in to an adjacency matrix 
#adj = gs.util.numpy_to_adjacency(network)
# build a model from make_model 
M = gs.GRNmodel.make_model(network)
for i in vars(M.parameters).keys():
    if i != "time":
        print(i,vars(M.parameters)[i])
M.set_pert("diag",effect=(0.5,0.9),noise=0,multipert=3,exp=1,reps=3,sign=-1)

#for i in vars(M.parameters):
#    if i != "time":
#        print(i,eval("M.parameters."+i))

datasets = pd.DataFrame()
noisesets = pd.DataFrame()
nfdatasets = pd.DataFrame()
print(M.model)
pos = 0
fig = 1
for s in [10]:
    print("Noise",s)
    for i in range(1):
        M.simulate_data(exp_type="ss",SNR=s,noise_model="normal")
    
        noise = M.noise
        data = M.data
        noise_free_data = M.noise_free_data
        print(M.noise_free_data)
        #print(M.noise)
        print(M.steady_state_RNA)
        fold_change_data = np.zeros((genes,genes*3))
        for i in range(genes*3):
            for j in range(genes):
                fold_change_data[j,i] = np.log2(M.noise_free_data.values[j,i]/M.steady_state_RNA[j])
        fold_change_data = pd.DataFrame(fold_change_data,index=M.data.index,columns=M.data.columns)
        #np.log2(np.divide(M.noise_free_data,np.reshape(np.array(M.steady_state_RNA),(-1,1))))
        print(fold_change_data)
        #print(M.model)
        #lscon = infer_networks(fold_change_data,M.perturbation,"lscon",np.logspace(-6,0,100))
        #correct = gs.benchmarking.compare_networks(lscon,M.network,exclude_diagonal=True)
        #print(correct["AUPR"],correct["AUROC"])
        #gs.analysis.network_data_dyn_test(M)

fig = plt.figure(constrained_layout=True)
(subfig1, subfig2,subfig3) = fig.subfigures(3, 1)

ax1 = subfig1.subplots(1,1)
ax2 = subfig2.subplots(1,1)
ax3 = subfig3.subplots(1,1)


G = nx.from_pandas_adjacency(network,create_using=nx.DiGraph)
edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
colors = list()
for w in weights:
    if w < 0:
        colors.append("r")
    else:
        colors.append("b")
try:
    nx.draw_planar(G,with_labels=True,ax=ax1,edgelist=edges,edge_color=colors)
except nx.NetworkXException: 
    nx.draw_spring(G,with_labels=True,ax=ax1,edgelist=edges,edge_color=colors)

if genes <= 20:
    sns.heatmap(M.noise_free_data.iloc[:,0:genes],cmap="vlag",linecolor="k",linewidths=0.01,ax=ax2,annot=True)
    sns.heatmap(fold_change_data.iloc[:,0:genes],cmap="vlag",linecolor="k",linewidths=0.01,ax=ax3,annot=True)
else:
    sns.heatmap(M.noise_free_data.iloc[:,0:genes],cmap="vlag",linecolor="k",linewidths=0.01,ax=ax2)
    sns.heatmap(fold_change_data.iloc[:,0:genes],cmap="vlag",linecolor="k",linewidths=0.01,ax=ax3)

    
subfig1.suptitle("GRN")
subfig2.suptitle("Noise free data")
subfig3.suptitle("Noise free fold change")

plt.show()
plt.close()
#sns.heatmap(M.data,cmap="YlGn",linecolor="k",linewidths=0.01)
sns.heatmap(fold_change_data.iloc[:,0:genes],cmap="vlag",linecolor="k",linewidths=0.01)
plt.show()
#sns.heatmap(M.data-M.noise_free_data,cmap="YlOrRd",linecolor="k",linewidths=0.01)
#plt.show()

#M.save("test_GS_GRNmodel.pickle")

#loadmodel = gs.GRNmodel.load("test_GS_GRNmodel.pickle")
#for key in loadmodel.__dict__.keys():
#    if key == "parameters":
#        print(key,loadmodel.__dict__[key].__dict__)
#    else:
#        print(key,loadmodel.__dict__[key])
