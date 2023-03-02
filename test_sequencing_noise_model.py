import array

import numpy as np
import pandas as pd
from methods.grn_inference import generate_sequencing_noise

genes_ind = [100, 101, 200, 201, 300, 301, 1000, 1001, 2000, 2001,
             3000, 3001, 10100, 10101, 10200, 10201, 10300, 10301, 11000, 12000] # static
#genes_ind = np.random.randint(1, len(g_length), size=ng) # random

# 5 is the optimal coverege for mRNA-seq (based on article cited in the draft)
# 50-75 is usually the length of read for mRNA-seq
# 2 technical replicates are usually done

# error rates

# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4631051/
# good mapping rates 0.1-0.2 (good) or 0-0.1 (very good)

nois_vgood = generate_sequencing_noise(20, 20, genes_ind, c=5, L=75, techrep=2, ermin=0, ermax=0.1)
print(nois_vgood["ratios"])
print(nois_vgood["snrs"])

nois_good = generate_sequencing_noise(20, 20, genes_ind, c=5, L=75, techrep=2, ermin=0.1, ermax=0.2)
print(nois_good["ratios"])
print(nois_good["snrs"])

# based on: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-13-185
# e.g. here for ENCODE project mismatch counting is 0.45-073
nois_encode_k562 = generate_sequencing_noise(20, 20, genes_ind, c=5, L=75, techrep=2, ermin=0.45, ermax=0.73)
print(nois_encode_k562["ratios"])
# SNR
print(nois_encode_k562["snrs"])

# similarities in depth, Fig.1 https://www.nature.com/articles/nrg3642

