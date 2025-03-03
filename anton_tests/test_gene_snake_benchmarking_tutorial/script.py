
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
	Y = Y.T, P = P.T, method = 'zscore')
# Benchmark. A high level interface that doesn't expose all options.
# Similar can be done manually like below
model_name = 'Tutorial_LSCON'

all_stats = gs.benchmarking.benchmark(
	estimated_network = estimated_network,
	true_network = true_network,
	output_dir = 'outputs',
	model_name = model_name,
	)
print(all_stats)


### Manual benchmarking
# Similar thing, but an interface that exposes
# more internals and options
cutoffs = gs.benchmarking.calculate_cutoffs(
	estimated_network,
	# num_cutoffs = 640,
	# all_cutoffs = True,
	# manual_cutoffs = True,
	# relative_cutoffs = np.logspace(-5, 0, 100)
	)
# Retain a copy of the original estimated network, since
# cut_network operates inplace to avoid repeatedly copying the network.
from copy import deepcopy
network = deepcopy(estimated_network)
tn = deepcopy(true_network)
ms = []
for cutoff in cutoffs:
	gs.benchmarking.cut_network(network, cutoff)
	# Warning:
	# compare_networks can modify both networks and true_network
	# This to avoid repeated copying of networks in the hot loop
	# of the benchmarking. Surrounding code has to be aware of
	# this and handle it.
	m = gs.benchmarking.compare_networks(
		networks = [network],
		true_network = tn,
		exclude_diag = True,
		include_sign = False)
	ms.append(m)
metrics = pd.concat(ms, axis = 0, ignore_index = True)

output_dir = Path('outputs2')
output_dir.mkdir(exist_ok = True)
metrics.to_csv(
	path_or_buf = output_dir / f'benchmark_stats_{model_name}.csv',
	sep = '\t', header = True, index = True)

stats = gs.benchmarking.calculate_auroc_and_aupr(
	full_network = estimated_network,
	true_network = true_network,
	metrics = metrics,
	)
gs.benchmarking.plot_auroc_and_aupr(
	stats = stats,
	model_name = model_name,
	output_dir = output_dir,
	fix_ylim = False)
epr = gs.benchmarking.calculate_dspin_stats.early_precision_rate(
	inferred_network = estimated_network,
	true_network = true_network,
	fraction_of_K = 1)
top_k_accuracy = gs.benchmarking.calculate_dspin_stats.top_k_accuracy(
	inferred_network = estimated_network,
	true_network = true_network,
	fraction_of_K = 1)
stats.pop('plot_metrics')
print({**stats, 'EPR': epr, 'top_K_acc': top_k_accuracy})
