"""
A collection of functions that are used to generate noise for the GRNmodle.py script.
Note that they are not imported in to the data namespace only the model class. 
"""

from pathlib import Path
import statistics as st
from numpy import inf
import pandas as pd
import numpy as np
import random
import math


def zinbme(data, prob_nb, sf): # sf is scaling factor (SNR like)
    n = round(np.mean(data)+1)
    print(n)
    nb = np.random.negative_binomial(n, prob_nb, size=len(data))  # negative binomial distribution
    zinb = nb
    msf = np.sqrt(np.var(data) / (sf * (np.var(zinb))))  # scaling factor to adjust noise values to data
    return pd.Series(zinb*msf)


class noise_gen:
    
    
    def make_gaussian(data,SNR):
        """
        Function that wraps the numpy.random.normal function for easy use with the geneSNAKE GNRmodel data generation 
        for adding white noise to experiemnts. 
        INPUT: 
            data - numpy.array, the data to add noise to. This is used both to get magnitude of noise and size of noise matrix  
            SNR - int or float, the requested SNR. This value will be used to determine the standard deviation needed to be added 
                  to get approximately that SNR in the output data. 
        OUTPUT: 
            noise - a numpy array of size data with the data+noise. 
        """
        
        # then we use the SNR equation to find the stdE based on the relation SNR = (mean(data))/stdE(noise)
        # variance(noise) = stdE(noise)**2
        # take this as absolute value as data where all values are extremely close to 0 can generate negative values
        # which will crash things
        stdE = np.abs(np.mean(data)/SNR)
        noise = np.random.normal(0,stdE,data.shape)       
        
        data = data+noise
        mask = data<=0
        mask = mask*np.random.uniform(0,min(1e-5,np.min(abs(data))),data.shape)
        
        data[np.where(data<=0)] = 0
        data = data+mask
        
        return data, noise
    
    
    def make_micro_array_like(data,SNR):
        """
        Model for creating micro array like noise based on the implementation of a similar nature in the GeneNetWeaver paper and the 
        Statistical analysis of MPSS measurements: Application to the study of LPS-activated macrophage gene expression paper by 
        Stolovitzky et al 2005. 
        Note that the output SNR will not be in line with the requested SNR but they will depend on one another in a way such that 
        a lower SNR will lead to higher noise. 
        INPUT:
            data - numpy.array, the data to add noise to. This is used both to get magnitude of noise and size of noise matrix 
            SNR - int or float, the requested SNR. This value will be used to determine the standard deviation needed to be added 
                  to get approximately that SNR in the output data. 
        OUTPUT: 
            noise - a numpy array of size data with the data+noise. 
        """
        
        # create a vectorized function that will be applied to each cell of the numpy array seperatly
        # the noise model is built around finding a variance value dependent on x and multiplying x with this.
        # meaning that small x values will have low noise and large x values have larger noise. 
        f = lambda x: x*np.exp(np.random.normal(0,x/SNR))
        # then feed the data in to this function 
        noise = f(data)
        
        return noise
    
    
    def make_sequencing_noise(data, SNR, genes_ind="random",  c=5, L=75, techrep=2, ermin=0.45, ermax=0.73, prob_zi=0.1, prob_nb=0.5, sf=0.05):
        """
        Function for generating sequencing-based noise based on coverage, GC-content and sequencing error rates
        INPUT:
            ng - number of genes
            nsamp - number of samples
            genes_ind - indices of genes
            c - coverage
            L - read length [bp]
            techrep - number of technical replicates
            ermin - minimum sequencing error rate, 0-1
            ermax - maximum sequencing error rate, 0-1
            prob_zi - probability of zero-inflation, 0-1
            prob_nb - probability of noiseless data from negative binomial distribution, e.g. value around 1 is lack of noise
            sf - scaling factor that adjusts variance in data to variance in noise
        # default settings based on: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-13-185
        # e.g. here for ENCODE project mismatch counting is 0.45-073
                
        """
        nsamp = np.shape(data)[1]
        ng = np.shape(data)[0]
        if SNR == "very_low":
            ermin = 0.7
            ermax = 1
            prob_zi = 0.4
            prob_nb = 0.1
            sf = 0.005
        elif SNR == "low":
            ermin = 0.6
            ermax = 0.9
            prob_zi = 0.3
            prob_nb = 0.1
            sf = 0.01
        elif SNR == "medium":
            ermin = 0.45
            ermax = 0.73
            prob_zi = 0.1
            prob_nb = 0.5
            sf = 0.05
        elif SNR == "high":
            ermin = 0.1
            ermax = 0.2
            prob_zi = 0.1
            prob_nb = 0.5
            sf = 0.1
        elif SNR == "very_high":
            ermin = 0
            ermax = 0.1
            prob_zi = 0.05
            prob_nb = 0.7
            sf = 0.5
        
        
        # code block 3
        p = Path(__file__).with_name("pc_genes_start_end_gc.txt")
        df = pd.read_csv(p, sep='\t', header=0)
        
        # genes lengths
        g_length = df["end_position"]-df["start_position"]
        # genes gc content
        gc_cont = df["percentage_gene_gc_content"]
        
        # randomize genes
        rand_genes_ind = genes_ind
        if genes_ind == "random":
            rand_genes_ind = np.random.randint(1, len(g_length), size=ng)
        else:
            rand_genes_ind = genes_ind
            
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
        frag = np.arange(0,G) # fragment of sequenced genome
        g_length_set = g_length[rand_genes_ind] # subset of genes lengths
        
        for t in range(0, techrep):
            allCounts = []
            allNoErrorCounts = []
            for s in range(0, nsamp):
                # error rate of reads
                # based on: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-13-185
                # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4631051/
                er = random.uniform(ermin, ermax)  # good mapping rates 0.1-0.2 (good) or 0-0.1 (very good)
                # e.g. here for ENCODE project mismatch counting is 0.45-073
                # https: // bmcbioinformatics.biomedcentral.com / articles / 10.1186 / 1471 - 2105 - 13 - 185 / tables / 3
                Ne = round(N - N * er)  # we decrease the number of reads as some of them are mismatched with a given error rate
                reads_pos = np.random.choice(frag, p=prbs, size=Ne, replace=True)
                counts_samp = []
                allCounts.append([])
                allNoErrorCounts.append([])
                tmp_start = 0
                for n in range(0, len(g_length_set)):  # loop over genes
                    tmp_len = g_length_set.iloc[n]  # tmp variables are to slide over all genes and check how many reads are mapped
                    tmp_end = tmp_start + tmp_len
                    tmp_counts = (reads_pos > int(tmp_start + L)) & (reads_pos < int(tmp_end - L))
                    tmp_start = tmp_end
                    counts_samp.append(sum(tmp_counts))
                counts = pd.Series(counts_samp)
                necounts = pd.Series(c * g_length_set / L)
                
                allCounts[s].append(np.transpose(counts))
                allNoErrorCounts[s].append(np.transpose(necounts))
            out_counts = np.transpose(pd.DataFrame(np.concatenate(allCounts)))  # samples in columns, genes in rows
            out_necounts = np.transpose(pd.DataFrame(np.concatenate(allNoErrorCounts)))  # samples in columns, genes in rows
        out_counts += out_counts
        out_necounts += out_necounts
        # zero inflation negative binomial noise to genes
        # zero inflation simulates dropouts
        # negative binomial simulates noise strength in counts
        
        out_counts = pd.DataFrame(round(out_counts/techrep))
        out_error = out_counts.apply(lambda x: zinbme(x, prob_nb, sf), axis=1)
        
        ratios = np.divide(out_counts+out_error, out_necounts / techrep)
        ratios = ratios.apply(lambda x: np.multiply(x, np.random.choice([0, 1], p=[prob_zi, 1 - prob_zi], size=len(x), replace=True)), axis=1)
        ratios[ratios == inf] = 0
        out_counts[out_counts == inf] = 0
        op = dict()
        op['ratios'] = ratios
        op['seq_counts'] = round(out_counts)
        op['noerror_counts'] = round(out_necounts / techrep)
        op['zinb_counts'] = out_error
        
        return op
