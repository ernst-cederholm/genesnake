
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
# An example GRN inference with normalized least squares regression (LSCON)
# Hillerton et al. 2022
aest = gs.inference.infer_networks(Y = Y, P = P, method = 'lscon')

# Benchmark. A high level interface that doesn't expose all options.
# Similar can be done manually like below
gs.benchmarking.benchmark(
    estimated_network = aest,
    true_network = M.network,
    output_dir = 'outputs',
    model_name = 'Tutorial_LSCON',
    )



# Manual benchmarking. An interface that exposes more options
inets = gs.benchmarking.cutoff_handler(aest, zvec_auto = True)
out = gs.benchmarking.compare_networks(inets, M.network, exclude_diag=False, include_sign=False)
outm = out['metrics']

### Save results to disk
output_dir = Path('outputs2')
[d.mkdir(exist_ok = True) for d in (output_dir,)]
model_name = 'Tutorial_LSCON'
outm.to_csv(
    path_or_buf = output_dir / f'benchmark_stats_{model_name}.csv', 
    sep = '\t', header = True, index = True)

print(
    gs.benchmarking.auroc_and_aupr(
        full_network = aest,
        true_network = M.network,
        metrics = outm,
        model_name = model_name,
        output_dir = output_dir,
        fix_ylim = False)
    )
