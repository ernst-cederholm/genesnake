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
    inets = gs.benchmarking.cutoff_handler(aest, zvec_auto = True) # create a span of networks from full to empty

    M_network = M.network
    anton_util.pickle_object((M_network, inets, aest), cache_file)

else:
    M_network, inets, aest = anton_util.unpickle_object(cache_file)



out = gs.benchmarking.compare_networks(inets, M_network, exclude_diag=False, include_sign=False)

model_name = 'Tutorial_LSCON'

outm = out['metrics']
outm.to_csv(
    path_or_buf = output_dir / f'benchmark_stats_{model_name}.csv', 
    sep = '\t', header = True, index = True)



gs.benchmarking.auroc_and_aupr(
    full_network = aest,
    true_network = M_network,
    metrics = outm,
    model_name = model_name,
    output_dir = output_dir)







