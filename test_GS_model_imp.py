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

# a cheat way to use a random or set network
A = 0
if A>0:
    # make a pre defeind network
    genes = 5
    network = np.array([[0,0,0,0,0],[-1,0,0,0,0],[0,0,0,1,0],[0,1,0,0,0],[0,0,0,1,0]])
    network = pd.DataFrame(network)
    network.columns=["G"+str(i+1) for i in range(genes)]
    network.index = network.columns
elif A<0:
    genes = 8
    network = np.array([[0,1,0,0,0,0,0,0],
                        [0,0,1,0,0,0,0,0],
                        [0,0,0,1,0,1,0,0],
                        [-1,0,0,0,0,-1,0,0],
                        [0,0,0,0,0,0,0,0],
                        [-1,0,0,-1,0,1,1,0],
                        [0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,1,1]])
    network = pd.DataFrame(network)
    network.columns=["G"+str(i+1) for i in range(genes)]
    network.index = network.columns
else:
    genes = int(sys.argv[1])
    # or creat a random one using the gs functions
    #network = gs.grn.make_FFLATnetwork(NETWORK_SIZE=genes,FFL_ENRICHED=0,self_loops=True)
    network = gs.grn.make_DAG(genes,1.2,sign_bias=0.5)

# build a model from make_model
M = gs.GRNmodel.make_model(network,combinations={"comb_prob":0.5})

set_param = 0
if set_param>0:
    for key in M.parameters.alphas.keys():
        if M.parameters.alphas[key]!=0 and len(key.split("_"))>1:
            M.parameters.alphas[key]=1
        elif M.parameters.alphas[key]!=0:
            M.parameters.alphas[key]=0.1

        M.parameters.hill_coef = [4 for _ in range(genes)]
        M.parameters.dis_const = {key:0.8 for key in M.parameters.dis_const.keys()}
        M.parameters.lambda_RNA = [0.8 for _ in range(genes)]
        M.parameters.lambda_prot = [0.6 for _ in range(genes)]

for i in vars(M.parameters).keys():
    if i != "time":
        print(i,vars(M.parameters)[i])
M.set_pert("diag",effect=(0.89,0.91),noise=0,multipert=3,exp=1,reps=3,sign=-1)

#for i in vars(M.parameters):
#    if i != "time":
#        print(i,eval("M.parameters."+i))

datasets = pd.DataFrame()
noisesets = pd.DataFrame()
nfdatasets = pd.DataFrame()
print(network)
print(M.model)
pos = 0
fig = 1
for s in [10]:
    print("Noise",s)
    for i in range(1):
        M.simulate_data(exp_type="ss",time_points=8,noise_model="normal")

        noise = M.noise
        data = M.data
        noise_free_data = M.noise_free_data
        M.noise_free_data.to_csv("data_test.csv")
        print(M.noise_free_data)
        #print(M.noise)
        print(M.steady_state_RNA)
        print(M.steady_state_prot)
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
(subfig1,subfig2, subfig3,subfig4) = fig.subfigures(4, 1)

ax1 = subfig1.subplots(1,1)
ax4 = subfig2.subplots(1,1)
ax2 = subfig3.subplots(1,1)
ax3 = subfig4.subplots(1,1)


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
    sns.heatmap(pd.DataFrame(M.steady_state_RNA).T,cmap="vlag",linecolor="k",linewidths=0.01,ax=ax4,annot=True)
    sns.heatmap(M.noise_free_data.iloc[:,0:genes],cmap="vlag",linecolor="k",linewidths=0.01,ax=ax2,annot=True)
    sns.heatmap(fold_change_data.iloc[:,0:genes],cmap="vlag",linecolor="k",linewidths=0.01,ax=ax3,annot=True)
else:
    sns.heatmap(M.steady_state_RNA,cmap="vlag",linecolor="k",linewidths=0.01,ax=ax4,annot=True)
    sns.heatmap(M.noise_free_data.iloc[:,0:genes],cmap="vlag",linecolor="k",linewidths=0.01,ax=ax2)
    sns.heatmap(fold_change_data.iloc[:,0:genes],cmap="vlag",linecolor="k",linewidths=0.01,ax=ax3)


subfig1.suptitle("GRN")
subfig2.suptitle("inital steady_state_RNA")
subfig3.suptitle("Noise free data")
subfig4.suptitle("Noise free fold change")

plt.show()
plt.close()
#sns.heatmap(M.data,cmap="YlGn",linecolor="k",linewidths=0.01)
#sns.heatmap(fold_change_data.iloc[:,0:genes],cmap="vlag",linecolor="k",linewidths=0.01)
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
