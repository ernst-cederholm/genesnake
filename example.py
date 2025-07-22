import numpy as np 
import pandas as pd
import genesnake as gs



model_type = 'lsco'
genes = 100
self_loops = True
cell_replicates = 3
snr = 3
noise_model = 'normal_ab_mod'
using_fold_change_data = True


network = gs.grn.make_FFLATTnetwork(
    network_size = genes, sparsity = 3, 
    self_loops = self_loops
    )

# anton_util.log_timestamp('Network made')
M = gs.GRNmodel.make_model(network)
# anton_util.log_timestamp('Model made')
M.set_pert(
    'diag', effect = (0.9, 1), noise = 0.01, 
    reps = cell_replicates, sign = -0.1
    )

# anton_util.log_timestamp('Perturbations set')
M.simulate_data(exp_type = 'ss', SNR = snr, noise_model = noise_model)
# anton_util.log_timestamp('Data simulated')
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

