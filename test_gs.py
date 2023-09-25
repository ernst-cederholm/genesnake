# let's start with importing GeneSnake
import genesnake as gs
from methods.grn_inference import infer_networks, compare_networks
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# first we create a random network of 10 genes and outdegree 2
# netf = gs.grn.make_FFLATnetwork()
ngenes = 20
#net = gs.grn.make_random(ngenes, 3)
#net = gs.grn.make_FFLATnetwork(ngenes)
net = gs.grn.make_DAG(ngenes, 3, self_loops=False)
# then build an ODE model based on network
M = gs.GRNmodel.make_model(net)

# set up parameters to default
M.parameters.generateMP()
# generate perturbation design for our data
# with "diag" each gene is perturbed once per replicate*experiemnts
# the strength of perturbation is between 0.6-0.8, and no noise
# in this scenario, exp=1 is a single experiment per gene where reps=2 define two replicates
# sign determines if perturbation means up or downregulation, here sign=-1 defines downregulation
M.set_pert("diag", effect=(0.9, 1), noise=0, exp=1, reps=2, sign=-1)

# run simulation of gene expression data
# we set experiment type to steady state(ss) and data space to linear (lin) or logarithmic (log)
# here the noise is Gaussian
#M.simulate_data(exp_type="ss", draw_pattern="lin", time_points=10, noise_model="gaussian", SNR=100)
M.simulate_data(exp_type="ss", draw_pattern="lin", time_points=100, noise_model="normal", SNR=10)
# our data is then stored under M.data
print(M.data)

# but nosie free data are kept as well
#print(M.noise_free_data)

# infer a network
# sample P matrix (sP) 4x8
P1 = np.eye(ngenes, ngenes)  # create matrix with 1's on diagonal (for 1 replicate)
P1 = pd.DataFrame(P1)
P = -pd.concat([P1, P1], axis=1)  # as two replicates

zvec = np.logspace(-6, 1, 30)

nets = infer_networks(M.noise_free_data, P, "rbfsvm", zvec)
print(nets)

out = compare_networks(nets, net, exclude_diagonal=False, signed=False)

print(out['AUROC'])
print(max(out['metrics']['f1']))

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2)
fig.subplots_adjust(wspace=0.01)
sns.heatmap(net.transpose(), cmap="PuRd", linecolor="k", ax=ax1, linewidths=0.01)
sns.heatmap(M.noise_free_data, cmap="YlGn", linecolor="k", ax=ax2, linewidths=0.01)
sns.heatmap(M.data, cmap="YlGn", linecolor="k", ax=ax3, linewidths=0.01)
sns.heatmap(M.data-M.noise_free_data, cmap="YlOrRd", linecolor="k", ax=ax4, linewidths=0.01)
fig.subplots_adjust(wspace=0.001)
ax1.title.set_text('Network')
ax2.title.set_text('Data')
ax3.title.set_text('Data+Noise')
ax4.title.set_text('Noise')
plt.show()


#gs.GRNmodel.save(M.model, file='C:\\Users\\mateu\\Desktop\\model.json', kind='json')

# save the output as pickle
# please note that this is Python-specific format so non-Python programs may not be able to read it
# file can be set to a full path and/or name of the file
# M.save(file='model.pickle', kind='pickle')
