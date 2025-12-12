import numpy as np 
import pandas as pd
import genesnake as gs


# Parameters

model_type = 'lsco'
genes = 100
cell_replicates = 3
snr = 3
noise_model = 'normal'
using_fold_change_data = True





# Generating network, generating data, inferring, and benchmarking

network = gs.grn.make_FFLATTnetwork(network_size = genes, sparsity = 3)

M = gs.GRNmodel.make_model(network)
M.set_pert(
	'diag', effect = (0.9, 1), noise = 0.01, 
	reps = cell_replicates, sign = -0.1
	)

M.simulate_data(exp_type = 'ss', SNR = snr, noise_model = noise_model)
noise = M.noise
data = M.data
noise_free_data = M.noise_free_data
reference_network = M.network

if using_fold_change_data:
	fold_change_data = np.zeros((genes,genes*cell_replicates))
	nsf_fold_change_data = np.zeros((genes,genes*cell_replicates))
	for i in range(genes*cell_replicates):
		for j in range(genes):
			fold_change_data[j,i] = np.log2(M.data.values[j,i]/M.steady_state_RNA[j])
			nsf_fold_change_data[j,i] = np.log2(M.noise_free_data.values[j,i]/M.noise_free_steady_state_RNA[j])
	fold_change_data = pd.DataFrame(fold_change_data,index=M.data.index,columns=M.data.columns)
	nsf_fold_change_data = pd.DataFrame(nsf_fold_change_data,index=M.data.index,columns=M.data.columns)
	expression_data = fold_change_data
else:
	expression_data = M.data

from copy import deepcopy
expression_data_copy = deepcopy(expression_data)

estimated_network = gs.inference.infer_networks(
	Y = expression_data.T, 
	P = M.perturbation.T,
	method = model_type)

all_stats = gs.benchmarking.benchmark(
	estimated_network = estimated_network,
	reference_network = reference_network.astype(bool),
	plot_dir = 'example_outputs',
	method_name = 'example',
	exclude_diagonal = True
	)

print(f'{all_stats = }')






# Plotting network and data

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

genes = 10  # Smaller, for greater clarity

network = gs.grn.make_FFLATTnetwork(network_size = genes, sparsity = 3)

M = gs.GRNmodel.make_model(network)
M.set_pert(
	'diag', effect = (0.9, 1), noise = 0.01, 
	reps = cell_replicates, sign = -0.1
	)

M.simulate_data(exp_type = 'ss', SNR = snr, noise_model = noise_model)
noise = M.noise
data = M.data
noise_free_data = M.noise_free_data
reference_network = M.network

if using_fold_change_data:
	fold_change_data = np.zeros((genes,genes*cell_replicates))
	nsf_fold_change_data = np.zeros((genes,genes*cell_replicates))
	for i in range(genes*cell_replicates):
		for j in range(genes):
			fold_change_data[j,i] = np.log2(M.data.values[j,i]/M.steady_state_RNA[j])
			nsf_fold_change_data[j,i] = np.log2(M.noise_free_data.values[j,i]/M.noise_free_steady_state_RNA[j])
	fold_change_data = pd.DataFrame(fold_change_data,index=M.data.index,columns=M.data.columns)
	nsf_fold_change_data = pd.DataFrame(nsf_fold_change_data,index=M.data.index,columns=M.data.columns)
	expression_data = fold_change_data
else:
	expression_data = M.data




fig = plt.figure(figsize = (6.4, 12), constrained_layout=True)
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

sns.heatmap(pd.DataFrame(M.steady_state_RNA).T,cmap="vlag",linecolor="k",linewidths=0.01,ax=ax4,annot=True)
sns.heatmap(M.noise_free_data.iloc[:,0:genes],cmap="vlag",linecolor="k",linewidths=0.01,ax=ax2,annot=True)
sns.heatmap(fold_change_data.iloc[:,0:genes],cmap="vlag",linecolor="k",linewidths=0.01,ax=ax3,annot=True)
	
subfig1.suptitle("GRN")
subfig2.suptitle("inital steady_state_RNA")
subfig3.suptitle("Noise free data")
subfig4.suptitle("Fold change data")

fig.savefig('example_outputs/network_and_data.svg')


print()
print('Output figures written to ./example_outputs')
