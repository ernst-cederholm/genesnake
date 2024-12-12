
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
estimated_network = gs.inference.infer_networks(Y = Y, P = P, method = 'lscon')
estimated_network.columns = estimated_network.index
# Benchmark. A high level interface that doesn't expose all options.
# Similar can be done manually like below
model_name = 'Tutorial_LSCON'
gs.benchmarking.benchmark(
	estimated_network = estimated_network,
	true_network = true_network,
	output_dir = 'outputs',
	model_name = model_name,
	)



# Manual benchmarking. Similar thing, but an interface that exposes
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
ms = []
for cutoff in cutoffs:
	gs.benchmarking.cut_network(network, cutoff)
	m = gs.benchmarking.compare_networks([network], true_network)
	ms.append(m)
metrics = pd.concat(ms, axis = 0, ignore_index = True)

output_dir = Path('outputs2')
output_dir.mkdir(exist_ok = True)
metrics.to_csv(
	path_or_buf = output_dir / f'benchmark_stats_{model_name}.csv',
	sep = '\t', header = True, index = True)
print(
	gs.benchmarking.auroc_and_aupr(
		full_network = estimated_network,
		true_network = true_network,
		metrics = metrics,
		model_name = model_name,
		output_dir = output_dir,
		fix_ylim = False)
	)
