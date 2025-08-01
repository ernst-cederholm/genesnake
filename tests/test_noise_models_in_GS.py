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

network = gs.grn.make_FFLATnetwork(NETWORK_SIZE=genes)
#network = gs.grn.make_random(genes,2)
# turn that in to an adjacency matrix 
#adj = gs.util.numpy_to_adjacency(network)
# build a model from make_model 
M = gs.GRNmodel.make_model(network)
M.parameters.generateMP()
M.set_pert("diag",effect=(0.8,0.9),noise=0,multipert=3,exp=1,reps=3,sign=-1)

datasets = pd.DataFrame()
noisesets = pd.DataFrame()
nfdatasets = pd.DataFrame()

#fig,ax = plt.subplots(3,3)

fig = plt.figure(constrained_layout=True)
(subfig1, subfig2,subfig3) = fig.subfigures(3, 1)
(ax1,ax2,ax3) = subfig1.subplots(1, 3)
(ax4,ax5,ax6) = subfig2.subplots(1, 3)
(ax7,ax8,ax9) = subfig3.subplots(1, 3)

pos = 0
fig = 1
for s in [1,10,100]:
    print("Noise",s)
    for i in range(10):
        M.simulate_data(exp_type="ss",SNR=s,draw_pattern="lin",time_points=100,noise_model="normal")
    
        noise = M.noise
        noisesets["run:"+str(i+1)] = noise.melt()["value"]
        data = M.data
        datasets["run:"+str(i+1)] = data.melt()["value"]
        noise_free_data = M.noise_free_data
        nfdatasets["run:"+str(i+1)] = noise_free_data.melt()["value"]
        #print(M.noise_free_data)
        #print(M.noise)
        #print(M.data)
        #print(M.steady_state_RNA)
        fold_change_data = np.zeros((genes,genes*3))
        for i in range(genes*3):
            for j in range(genes):
                fold_change_data[j,i] = np.log2(M.data.values[j,i]/M.steady_state_RNA[j])
        
        #np.log2(np.divide(M.noise_free_data,np.reshape(np.array(M.steady_state_RNA),(-1,1))))
        #print(fold_change_data)
        lscon = infer_networks(fold_change_data,M.perturbation,"lscon",np.logspace(-6,0,100))
        correct = gs.benchmarking.compare_networks(lscon,M.network,exclude_diagonal=True)
        print(correct["AUPR"],correct["AUROC"])
        
    eval("subfig"+str(fig)).suptitle("SNR:"+str(s))
    fig+=1
    pos+=1
    sns.boxplot(noisesets,ax=eval("ax"+str(pos)))
    eval("ax"+str(pos)).set_title("Noise")
    pos+=1
    sns.boxplot(datasets,ax=eval("ax"+str(pos)))
    eval("ax"+str(pos)).set_title("Data+Noise")
    pos+=1
    sns.boxplot(nfdatasets,ax=eval("ax"+str(pos)))
    eval("ax"+str(pos)).set_title("Data")
    
M.simulate_data(exp_type="ss",SNR=s,draw_pattern="lin",time_points=100,noise_model="normal")
lscon = infer_networks(M.noise_free_data,M.perturbation,"lscon",np.logspace(-6,0,100))
correct = gs.benchmarking.compare_networks(lscon,M.network)
print("no noise")
print(correct["AUPR"],correct["AUROC"])
        
    
plt.show()
G = nx.from_pandas_adjacency(network,create_using=nx.DiGraph)
nx.draw(G,with_labels=True)
plt.show()
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
