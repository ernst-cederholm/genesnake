"""
Script that calculates the average rank of perturbation assuming perturbations should be negative 
"""

from scipy.stats import rankdata as rd
from scipy.stats import percentileofscore as pr
import numpy as np

def calc(data,pert_mat):
    """
    function for finding the relative rank of the perturbation and checking if it is on average higher or lower
    INPUT:
        data - a pandas.dataframe containing the values to be ranked 
        pert_mat - a dictonary containing the infomrmation of which row is perturbed in which column in the form {row:[col1,col2..coln]}
    OUTPUT:
        median_rank - average perturbation rank
        median_prank - average percentage rank of peturbed row
        times_smallest - how often is the perturbation rank 1 
        times_largest - how often is the pertrubation rank the largest value possible
    """
    # start by checking that the pert_mat contains row information
    # note that these will be in a string sort but that dosn't matter
    overlap = list(set(list(pert_mat.keys()))&set(list(data.index)))
    # make sure that at least 60% of the keys in the pertmat exist in the
    # row names 
    if len(overlap)<len(data.index)*0.6:
        print("\n\nWARNING:Row names in the perturbation matrix does not seem to match those in the expression data. Skipping perturbation rank calculations.\n\n")
        # return 3 Nones so that the snaked scripts don't crash 
        return(np.nan,np.nan,np.nan,np.nan)
    
    # get the ranked value of the dataframe
    data_rank = rd(data.values,axis=0,method='min')
    # get the percental increment value 
    pct_inc = 1./data.shape[0]
    
    # create two lists to store the values in for later averaging
    pert_ranks = list()
    perc_ranks = list()
    
    # loop over all indexes in the overlap list
    for ind in overlap:
        # get the row number for the current ind
        row = data.index.get_loc(ind)
        # and all columns for that indes
        for col in pert_mat[ind]:
            pert_ranks.append(data_rank[row,col])            
            percent = data_rank[:,col]*pct_inc
            perc_ranks.append(percent[row])
            
    median_rank = np.median(pert_ranks)
    median_prank = np.median(perc_ranks)
    # find the number of times it is the smallest value by finding all rank 1s 
    times_smallest = pert_ranks.count(1)
    # and the largest by finding all percent rank that are 1.0
    time_largest = perc_ranks.count(1.0)
    
    return(round(median_rank,4),round(median_prank,4),times_smallest,time_largest)
   
