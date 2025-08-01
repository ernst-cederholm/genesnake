"""
Script for checking if the dynamics in the system correlates with the sign in the network
Calculates a correlation between all genes and create a scatter plot of corr over network weight.  
"""
from scipy import stats as sts
import matplotlib.pyplot as plt
import genesnake as gs
import seaborn as sns
import networkx as nx
import pandas as pd
import numpy as np

# set number of genes
genes = 100
# creat a random one using the gs functions
#network = gs.grn.make_FFLATnetwork(NETWORK_SIZE=genes,FFL_ENRICHED=1,self_loops=False)
#network = pd.DataFrame(np.array([[0,0,0,0,0],[0.55,0,0,0,0],[0.38,0.6,0,0,0],[0,0.5,0,0,0],[0,0,0.3,0,0]]),index=["G1","G2","G3","G4","G5"],columns=["G1","G2","G3","G4","G5"])
network = gs.grn.make_DAG(genes,1,sign_bias=1)

# build a model from make_model 
M = gs.GRNmodel.make_model(network,combinations={"comb_prob":0})

# and a perturbation
M.set_pert("diag",effect=(0.89,0.91),noise=0,multipert=3,exp=1,reps=10,sign=-1)

#for key in M.parameters.alphas.keys():
    #if "_" in key:
        #if key.split("_")[0] == "G1":
        #M.parameters.alphas[key]=0.6
        #M.parameters.dis_const[key]=0.5

# run simulations 
M.simulate_data(exp_type="ss",SNR=10,noise_model="normal")

noise_free_data = M.noise_free_data.values

network_set = list()
corr_set = list()
names = list()

for i in range(genes):
    rows = np.where(M.network.values[i,:])[0].tolist()
    corr = sts.spearmanr(noise_free_data,axis=1)
    corr = corr.statistic
    for j in rows:
        corr_set.append(corr[i,j])
        network_set.append(M.network.values[i,j])
        names.append("G"+str(i+1)+",G"+str(j+1))

print(network)
print(M.steady_state_RNA)
print(M.noise_free_data)
print(M.perturbation)
for i in vars(M.parameters).keys():
    if i != "time":
        print(i,vars(M.parameters)[i])
print(M.model)
fig,ax = plt.subplots(2)
        
ax[0].scatter(network_set,corr_set,)

c=0
for i,j in zip(network_set,corr_set):
    ax[0].text(i,j,f'({names[c]})')
    c+=1
#plt.ylabel("Correlation between regulator and target")
#plt.xlabel("Network weight between regulator and target")

G = nx.from_pandas_adjacency(network,create_using=nx.DiGraph)
edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
colors = list()
for w in weights:
    if w < 0:
        colors.append("r")
    else:
        colors.append("b")

nx.draw_spring(G,with_labels=True,ax=ax[1],edgelist=edges,edge_color=colors)

plt.show()
