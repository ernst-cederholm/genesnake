
# This tests the benchmarking code with expression data originally from 
# Replogle et at 2022[1], target network from Beeline[2],
# and the dspin[3] method.

# Note that until I get around to figuring out
# Git Large File Storage
# you'll have to ask me for the input data for this one.

# More specifically, the target network is derived from the
# Non-specific-ChIP-seq-network.csv
# downloaded from Beelines data.
# The expression data is derived from the gemgroup Z-normalized 
# pseudo-bulk expression data from Replogle et al 2022

# The test case does not include all steps of the analysis for testing
# speed and convenience. Instead, both true and estimated networks
# are read from caches of filtered and harmonised networks.

# Get in touch if any questions!
# Anton Björk, 2024-12-11

# References:
# [1] Replogle et at 2022
	# Link
		# https://doi.org/10.1016/j.cell.2022.05.013
	# Citation
		# Replogle JM, Saunders RA, Pogson AN, Hussmann JA, Lenail A, Guna A, Mascibroda L, Wagner EJ, Adelman K, Lithwick-Yanai G, Iremadze N, Oberstrass F, Lipson D, Bonnar JL, Jost M, Norman TM, Weissman JS. Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq. Cell. 2022 Jul 7;185(14):2559-2575.e28. doi: 10.1016/j.cell.2022.05.013. Epub 2022 Jun 9. PMID: 35688146; PMCID: PMC9380471.
# [2] Beeline
	# Link
		# https://doi.org/10.1038/s41592-019-0690-6
	# Citation
		# Pratapa A, Jalihal AP, Law JN, Bharadwaj A, Murali TM. Benchmarking algorithms for gene regulatory network inference from single-cell transcriptomic data. Nat Methods. 2020 Feb;17(2):147-154. doi: 10.1038/s41592-019-0690-6. Epub 2020 Jan 6. PMID: 31907445; PMCID: PMC7098173.
# [3] D-SPIN
	# Link
		# https://doi.org/10.1101/2023.04.19.537364
	# Citation
		# D-SPIN constructs gene regulatory network models from multiplexed scRNA-seq data revealing organizing principles of cellular perturbation response
		# Jialong Jiang, Sisi Chen, Tiffany Tsou, Christopher S. McGinnis, Tahmineh Khazaei, Qin Zhu, Jong H. Park, Inna-Marie Strazhnik, Jost Vielmetter, Yingying Gong, John Hanna, Eric D. Chow, David A. Sivak, Zev J. Gartner, Matt Thomson
		# bioRxiv 2023.04.19.537364; doi: https://doi.org/10.1101/2023.04.19.537364

print('Imports')
from pathlib import Path
import pickle
import genesnake as gs

input_dir = Path('inputs')
output_dir = Path('outputs')
[d.mkdir(exist_ok = True) for d in (input_dir, output_dir)]

print('Main')
def main():

	print('Read data')
	sorted_networks_cache_path = input_dir / 'sorted_networks.pkl'
	with open(sorted_networks_cache_path, 'rb') as f:
		sorted_networks = pickle.load(f)
	true_network = sorted_networks.pop('true')

	print('Benchmark')
	model_name = 'dspin'
	estimated_network = sorted_networks[model_name]
	stats = gs.benchmarking.benchmark(
		estimated_network = estimated_network,
		true_network = true_network,
		output_dir = output_dir,
		model_name = model_name,
		)
	print(stats)

main()
