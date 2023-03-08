import statistics
import numpy as np
from methods.grn_inference import generate_sequencing_noise

genes_ind = [100, 101, 200, 201, 300, 301, 1000, 1001, 2000, 2001,
             3000, 3001, 10100, 10101, 10200, 10201, 10300, 10301, 11000, 12000] # static
# or random
#genes_ind = np.random.randint(1, len(g_length), size=ng) # random
# sample data
X = [1, 2, 10, 9, 50, 3, 6, 12, 37, 2, 1, 0, 10, 20, 1, 5, 9, 1, 0, 20]

# 5 is the optimal coverage for mRNA-seq (based on article cited in the draft)
# 50-75 is usually the length of read for mRNA-seq
# 2 technical replicates are usually done

# error rates

# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4631051/
# good mapping rates 0.1-0.2 (good) or 0-0.1 (very good)

# very high quality data
nois_vgq = generate_sequencing_noise(len(genes_ind), 10, genes_ind, c=5, L=75, techrep=2, ermin=0, ermax=0.1, prob_zi=0.05, prob_nb=0.7, sf=0.5)
Y = np.multiply(nois_vgq["ratios"][1], X)
E = Y - X
print("SNR very high quality")
print(statistics.variance(X) / statistics.variance(E))

# good high data
nois_gq = generate_sequencing_noise(len(genes_ind), 10, genes_ind, c=5, L=75, techrep=2, ermin=0.1, ermax=0.2, prob_zi=0.1, prob_nb=0.5, sf=0.1)
Y = np.multiply(nois_gq["ratios"][1], X)
E = Y - X
print("SNR high quality")
print(statistics.variance(X) / statistics.variance(E))

# based on: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-13-185
# e.g. here for ENCODE project mismatch counting is 0.45-073
# data
nois_k562 = generate_sequencing_noise(len(genes_ind), 10, genes_ind, c=5, L=75, techrep=2, ermin=0.45, ermax=0.73, prob_zi=0.1, prob_nb=0.5, sf=0.05)
Y = np.multiply(nois_k562["ratios"][1], X)
E = Y - X
print("SNR for K562, medium/low quality")
print(statistics.variance(X) / statistics.variance(E))

# similarities in depth, Fig.1 https://www.nature.com/articles/nrg3642

# very low quality data
nois_vgq = generate_sequencing_noise(len(genes_ind), 10, genes_ind, c=5, L=75, techrep=2, ermin=0.6, ermax=0.9, prob_zi=0.3, prob_nb=0.1, sf=0.01)
Y = np.multiply(nois_vgq["ratios"][1], X)
E = Y - X
print("SNR very low quality")
print(statistics.variance(X) / statistics.variance(E))