import genesnake as gs
from genesnake.analysis.snakeda.snakeda import snakeda as snakeda

# possible inputs: data, perts_df=None, norep=False, pert_design='single', out=None, direction="genes"

cell_replicates = 3
snr = 3
noise_model = 'normal'

genes = 15  # Smaller, for greater clarity

network = gs.grn.make_FFLATTnetwork(network_size = genes, sparsity = 3)

# default example
M = gs.GRNmodel.make_model(network)
M.set_pert(
    'diag', effect = (0.9, 1), noise = 0.01, 
	reps = cell_replicates)

M.simulate_data(exp_type = 'ss', SNR = snr, noise_model = noise_model)

print(snakeda(M))

# example with multiple perturbations per experiment
# output as json and html
M = gs.GRNmodel.make_model(network)
M.set_pert(
    'multi', effect = (0.9, 1), noise = 0.01, 
	reps = cell_replicates)

M.simulate_data(exp_type = 'ss', SNR = snr, noise_model = noise_model)

snakeda(M, pert_design='multiple', out="demo/snakeda_multi_pert")

# example with csv data, labels, replicates and no perturbation matrix
# output as html
snakeda("demo/snakeda_replicates.csv", out="demo/snakeda_replicates.html")

# example with csv data and seperate perturbation matrix without labels, transposed
# output as png
snakeda("demo/snakeda_transposed_nolabel.csv", perts_df="demo/snakeda_transposed_nolabel_p.csv", out="demo/snakeda_transposed_nolabel.png", direction="experiments")