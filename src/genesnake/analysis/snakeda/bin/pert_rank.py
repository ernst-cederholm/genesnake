from scipy.stats import rankdata
import numpy as np

def calc(data_df, rep_dict):
    """
    Function to calculate statistics related to the rank of perturbed genes.
    INPUT:
        data_df - matrix with expression data, genesXexperiments
        rep_dict - dictionary with gene names as keys and list of experiments where they are perturbed as values.
    OUTPUT:
        median_rank - median rank of perturbed gene in experiment
        median_p_rank - median rank of perturbed gene in experiment, expressed as a percentage
        times_smallest - number of times the perturbed gene ranked the lowest
        times_largest - number of times the perturbed gene ranked the highest
    """
    # check to ensure that enough of experiments are matched with which gene is perturbed, cutoff at 60%
    overlap = list(set(list(rep_dict.keys()))&set(list(data_df.index)))
    if len(overlap)<len(data_df.index)*0.6 or len(overlap)<len(rep_dict.keys())*0.6:
        print("\n\nWARNING:Row names in the perturbation matrix does not seem to match those in the expression data. Skipping perturbation rank calculations.\n\n")
        # return 4 nans so that the snaked scripts don't crash 
        return(np.nan,np.nan,np.nan,np.nan)

    # get the ranked values of the dataframe, rankes within columns
    data_rank = rankdata(data_df.values, axis=0, method='min')
    perc_rank = data_rank/data_df.shape[0]

    # lists to store values
    pert_ranks = list()
    pert_p_rank = list()

    # loop over lables in overlap list
    for name in overlap:
        # get the row index
        row = data_df.index.get_loc(name)
        # loop over experiments for given gene
        for col in rep_dict[name]:
            pert_ranks.append(data_rank[row, col])
            pert_p_rank.append(perc_rank[row, col])

    # calculate the median rank of perturbed genes
    median_rank = np.median(pert_ranks)
    median_p_rank = np.median(pert_p_rank)

    # find number of perturbations that ranks the lowest and highest for a given experiment
    times_smallest = pert_ranks.count(1)
    times_largest = pert_p_rank.count(1.0)

    return int(median_rank), median_p_rank, times_smallest, times_largest