import pandas as pd
import numpy as np
from numba import jit, prange

@jit(nopython=True, parallel=True)
def corr_calc(data, reps_list, norep):
    """
    Function for calculation the mean correlation of full dataset and the replicates within it.
    INPUT:
        data - numpy.array with data, genesXexperiments, sorted by experiment so all replicates of the same experiment are grouped
        reps_list - list of numpy.arrays where each array contain the column index of experiments that are replicates of eachother
    OUTPUT:
        mean_corr - mean correlation of columns over the whole dataset
        mean_rep_corr - mean correlation of replicates for an experiment 
    """
    n_rep_corrs = 0
    mean_rep_corr = 0
    n_tot_corrs = 0
    mean_corr = 0

    # loop over groups of replicates
    for i in prange(len(reps_list)):
        # correlation of replicates
        n_reps = len(reps_list[i])
        pearsons = data[:, reps_list[i]]
        pearsons = np.corrcoef(np.transpose(pearsons))
        mean = np.mean(pearsons[np.triu_indices(n_reps, k=1)]) * n_reps * (n_reps-1) / 2
        n_rep_corrs += n_reps * (n_reps-1) / 2
        mean_rep_corr += mean
        # loop over groups which has not been calculated correlation with
        for k in prange(i+1, len(reps_list)):
            pearsons = data[:, np.concatenate((reps_list[i], reps_list[k]))]
            pearsons = np.corrcoef(np.transpose(pearsons))
            mean = np.sum(pearsons[:len(reps_list[i]), len(reps_list[i]):])
            n_tot_corrs += len(reps_list[i])*len(reps_list[k])
            mean_corr += mean
    n_tot_corrs += n_rep_corrs
    mean_corr += mean_rep_corr

    if norep:
        mean_rep_corr = np.nan
    else:
        mean_rep_corr = mean_rep_corr/n_rep_corrs

    return mean_corr/n_tot_corrs, mean_rep_corr

def test(data_df, rep_dict, norep):
    """
    Wrapper function for calculating correlation related statistics of data set. Handles conversion of data so calculation function can be optimized.
    INPUT:
        data_df - pandas.DataFrame with values, genesXexperiments
        rep_dict - dictionary to lookup which columns are replicates of which experiment
        norep - option if there are not any replicates
    OUTPUT:
        mean_corr - mean correlation of columns over the whole dataset
        mean_rep_corr - mean correlation of replicates for an experiment
    """
    # convert input to work with numpy
    data = data_df.to_numpy()
    reps_list = [np.array(inds) for key, inds in rep_dict.items()]

    # run correlation calculations
    mean_corr, mean_rep_corr = corr_calc(data, reps_list, norep)

    return mean_corr, mean_rep_corr