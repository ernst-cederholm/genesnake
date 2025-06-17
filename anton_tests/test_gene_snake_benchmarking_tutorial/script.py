
import numpy as np
import pandas as pd
import genesnake as gs
from pathlib import Path

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
true_network = M.network
# An example GRN inference with normalized least squares regression (LSCON)
# Hillerton et al. 2022
estimated_network = gs.inference.infer_networks(
	Y = Y.T, P = P.T, method = 'zscore_ab')
# Benchmark. A high level interface that doesn't expose all options.
# Similar can be done manually like below
method_name = 'tutorial_dag_self_loops_true_zscore_ab'

# all_stats = gs.benchmarking.benchmark(
# 	estimated_network = estimated_network,
# 	true_network = true_network,
# 	output_dir = 'outputs',
# 	method_name = method_name,
# 	)

all_stats = gs.benchmarking.benchmark(
	estimated_network = estimated_network,
	reference_network = true_network.astype(bool),
	plot_dir = 'outputs',
	method_name = method_name,
	exclude_diagonal = False
	)
print(all_stats)
