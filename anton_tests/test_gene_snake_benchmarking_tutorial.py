import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
import genesnake as gs
from pathlib import Path



import anton_util


output_dir = Path('outputs')
cache_dir = output_dir / 'cache/'
[d.mkdir(exist_ok = True) for d in (output_dir, cache_dir)]




cache_file = cache_dir / 'cache.pkl'

if not cache_file.exists():

    ### Simulation
    genes = 50 # number of genes
    spars = 4 # sparsity

    network = gs.grn.make_DAG(genes, spars, self_loops=True)
    M = gs.GRNmodel.make_model(network)
    M.set_pert('diag', effect=(0.9, 1), noise=0.01, reps=3, sign=-0.1)
    M.simulate_data(exp_type='ss', SNR=10, noise_model='microarray')

    ### GRN inference
    Y = M.data # simulated expression data
    P = M.perturbation # perturbation (P) matrix

    # An example GRN inference with normalized least squares regression (LSCON)
    # Hillerton et al. 2022
    aest = gs.inference.infer_networks(Y = Y, P = P, method = 'lscon')

    ### Benchmarking
    inets = gs.benchmarking.cutoff_handler(aest) # create a span of networks from full to empty

    M_network = M.network
    anton_util.pickle_object((M_network, inets, aest), cache_file)

else:
    M_network, inets, aest = anton_util.unpickle_object(cache_file)



out = gs.benchmarking.compare_networks(inets, M_network, exclude_diag=False, include_sign=False)
# print all metrics as data frame
outm = out['metrics']
# print(outm)






### plotting part
# plot a boxplot with selected statistics
figure, axis = plt.subplots(1, 2)
pal1 = ['#f62681','#00ffff','#fef250','#ffa500','#c8a2c8']
bp = axis[0].boxplot(outm[["mcc","f1","ppv","fdr","tpr"]], labels=['MCC', 'F1', 'PPV', 'FDR', 'TPR'], patch_artist=True)
axis[0].grid(axis='y')

for patch, color in zip(bp['boxes'], pal1):
    patch.set_facecolor(color)

for median in bp['medians']:
    median.set(color='black',
               linewidth=3)

data = {'AUROC': out['AUROC'], 'AUPR': out['AUPR']}
courses = list(data.keys())
values = list(data.values())

# creating the bar plot of AUPR and AUC
axis[1].bar(courses, values, color=['#9bbb59','#38761d'], width=0.4)

for n, ax in enumerate(axis):
    ax.text(-0.05, 1.05, string.ascii_uppercase[n], transform=ax.transAxes,
            size=18, weight='bold')


plt.savefig(output_dir / 'metrics.svg')
# plt.show()








