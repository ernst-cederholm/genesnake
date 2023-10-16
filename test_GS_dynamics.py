"""
Script for checking if the dynamics in the system correlates with the sign in the network
Calculates a correlation between all genes and create a scatter plot of corr over network weight.  
"""
from scipy import stats as sts
import matplotlib.pyplot as plt
import genesnake as gs
import seaborn as sns
import pandas as pd
import numpy as np

# set number of genes
genes = 100
# creat a random one using the gs functions
#network = gs.grn.make_FFLATnetwork(NETWORK_SIZE=genes,FFL_ENRICHED=1,self_loops=False)
network = gs.grn.make_DAG(genes,1.2,sign_bias=0.7)

# build a model from make_model 
M = gs.GRNmodel.make_model(network,combinations={"comb_prob":0.0})

# and a perturbation
M.set_pert("diag",effect=(0.89,0.91),noise=0,multipert=3,exp=1,reps=10,sign=-1)

# run simulations 
M.simulate_data(exp_type="ss",SNR=10,noise_model="normal")

noise_free_data = M.noise_free_data.values

network_set = list()
corr_set = list()

for i in range(genes):
    rows = np.where(M.network.values[i,:])[0].tolist()
    corr = sts.spearmanr(noise_free_data,axis=1)
    corr = corr.statistic
    
    for j in rows:
        corr_set.append(corr[i,j])
        network_set.append(M.network.values[i,j])

plt.scatter(network_set,corr_set)

plt.ylabel("Correlation between regulator and target")
plt.xlabel("Network weight between regulator and target")
plt.show()
