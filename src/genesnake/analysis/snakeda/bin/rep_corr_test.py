import pandas as pd
import numpy as np

def test(data_df, rep_dict, norep):
    """
    Function for calculating correlation related statistics of data set.
    INPUT:
        data_df - pandas.DataFrame with values, genesXexperiments
        rep_dict - dictionary to lookup which columns are replicates of which experiment
        norep - option if there are not any replicates
    OUTPUT:
        median_corr - median correlation of columns over the whole dataset
        median_rep_corr - median correlation of replicates for an experiment
        num_greater_repcorrs - number of times the correlation of replicates is greater than the average correlation over the whole dataset
    """

    # calculate all pairwise correlations
    pearson_all = data_df.corr()
    median_corr = np.median(pearson_all.values[np.triu_indices(pearson_all.shape[0],k=1)])

    if not norep:
        reps_corrs = list()

        # pick out correlation scores for each group of replicates
        for key in rep_dict.keys():
            # make sure replicates exist
            if len(rep_dict[key])<2:
                pass
            else:
                # pick out correlation scores
                repcorr = pearson_all.iloc[rep_dict[key], rep_dict[key]]
                # pick only upper triangle without diagonal to merge into list
                repcorr = repcorr.values[np.triu_indices(repcorr.shape[0],k=1)]
                reps_corrs = [*reps_corrs, *repcorr]

        median_rep_corr = np.median(reps_corrs)

        # check how many replicate correlations are greater than the median correlation for the dataset
        num_reps = len(reps_corrs)
        greater_repcorrs = len([corr for corr in reps_corrs if corr > median_corr])
        greater_repcorrs = greater_repcorrs*100/num_reps
    else:
        median_rep_corr = np.nan
        greater_repcorrs = np.nan

    return median_corr, median_rep_corr, greater_repcorrs