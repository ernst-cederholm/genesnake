import random
import pandas as pd
import numpy as np
from pathlib import Path
def generate_sequencing_noise(ng, nsamp, genes_ind,  c=5, L=75, techrep=2, ermin=0, ermax=0.1):
    """
    Function for generating sequencing-based noise based on coverage, GC-content and sequencing error rates
    INPUT:
        ng - number of genes
        nsamp - numebr of samples
        genes_ind - indices of genes
        c - coverage
        L - read length [bp]
        techrep - number of technical replicates
        ermin - minimum sequencing error rate
        ermax - maximum sequencing error rate
    """

    p = Path(__file__).with_name("pc_genes_start_end_gc.txt")
    df = pd.read_csv(p, sep='\t', header=0)

    # genes lengths
    g_length = df["end_position"]-df["start_position"]
    # genes gc content
    gc_cont = df["percentage_gene_gc_content"]

    #number of genes
    #ng = 10  # number of genes
    #c = 1  # RNA-seq coverage
    #L = 75  # read length usually 50-75 bp for gene expression / RNA profiling
    #techrep = 2
    # randomize genes
    rand_genes_ind = genes_ind
    #rand_genes_ind = np.random.randint(1, len(g_length), size=ng)
    #rand_genes_ind = [1000, 2000, 10000, 10010, 5000, 7000, 7500, 100, 3500, 9000]

    g_start = df["start_position"][rand_genes_ind].to_frame()
    g_end = df["end_position"][rand_genes_ind].to_frame()
    # total length of the sequenced genome or genome fragment
    # in this code, it is a sum of all N genes lengths
    G = sum(g_length[rand_genes_ind])

    # now we use Lander-Waterman equation to estimate number of reads N for a given coverage
    N = round((c*G)/L)

    # randomize read positions from unif distribution
    sr = pd.Series(1-gc_cont[rand_genes_ind]/100)
    prbs = sr.repeat(g_length[rand_genes_ind])
    prbs /= prbs.sum()  # need to normalize it so it sums to 1
    frag = np.arange(0, G) # fragment of sequenced genome
    g_length_set = g_length[rand_genes_ind] # subset of genes lengths

    #nsamp = 10

    for t in range(0, techrep):
        allArrays = []
        all_snrs = []
        for s in range(0, nsamp):
            # error rate of reads
            # based on: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-13-185
            # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4631051/
            er = random.uniform(ermin, ermax) # good mapping rates 0.1-0.2 (good) or 0-0.1 (very good)
            # e.g. here for ENCODE project mismatch counting is 0.45-073
            # https: // bmcbioinformatics.biomedcentral.com / articles / 10.1186 / 1471 - 2105 - 13 - 185 / tables / 3
            Ne = round(N - N*er) # we decrease the number of reads as some of them are mismatched with a given error rate
            reads_pos = np.random.choice(frag, p=prbs, size=Ne, replace=True)
            counts_samp = []
            allArrays.append([])
            tmp_start = 0
            for n in range(0, len(g_length_set)):  # loop over genes
                tmp_len = g_length_set.iloc[n] # tmp variables are to slide over all genes and check how many reads are mapped
                tmp_end = tmp_start + tmp_len
                tmp_counts = (reads_pos > int(tmp_start+L)) & (reads_pos < int(tmp_end-L))
                tmp_start = tmp_end
                counts_samp.append(sum(tmp_counts))
            rate = np.divide(counts_samp, (c*g_length_set)/L) # estimated / expected
            snr = np.var((c * g_length_set) / L)/np.var(counts_samp - ((c * g_length_set) / L))  # var (signal) / var (noise)

            allArrays[s].append(np.transpose(rate))
            all_snrs.append(snr)
        out = np.transpose(pd.DataFrame(np.concatenate(allArrays)))  # samples in columns, genes in rows
    out += out

    op = dict()
    op['ratios'] = out/techrep
    op['snrs'] = all_snrs # snrs per sample

    return op