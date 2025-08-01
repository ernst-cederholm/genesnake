
from scipy.stats import ttest_ind as ttest
from scipy.stats import pearsonr as pear
from itertools import combinations
import pandas as pd
import numpy as np

def test(df,rep_mat):
    """
    Script for performing replicate based correlation analysis of data.
    INPUT:
        data - a pandas.dataframe containing the values to be ranked 
        pert_mat - a dictonary containing the infomrmation of which row is perturbed in which column in the form {row:[col1,col2..coln]}
    OUTPUT:
        median_repcorr - the median correlation between replicates
        median_corr - the median correlation for all columns
        greater_repcorrs - the number of correlation values for the replicates above the median_corr 
        sig_dif.pvalue - the statistical significance that the expected replicate correlations are greater than theexpected correlation value between all columns (here used as random correlation) 
    """
    
    # make sure df is numerical
    df = df.astype(float)
    
    # first get all pairwise correlations 
    pearson_all = [pear(x, y)[0] for x,y in combinations(np.transpose(df.values), 2)]
    median_corr = np.median(pearson_all)
    
    repcorr = list()
    
    # make a correlation score for each group in the rep_mat
    for key in rep_mat.keys():
        # make sure replicates exist
        if len(rep_mat[key]) < 2:
            repcorr.append(np.nan)
        else:
            # get the correlation for all replicates 
            rep_corr = df.iloc[:,rep_mat[key]].corr()
            # pick only the upper triangle without diagonal
            rep_corr = rep_corr.values[np.triu_indices(rep_corr.shape[0],k=1)]
            # and merge these in to the existing list
            repcorr = [*repcorr,*rep_corr]
            
    median_repcorr = np.median(repcorr)    
    
    greater_repcorrs = len(list(filter((median_corr).__le__,repcorr)))
    sig_dif = ttest(repcorr,pearson_all,equal_var=False,nan_policy='omit',alternative='greater')
    
    return round(median_repcorr,4),round(median_corr,4),greater_repcorrs,round(sig_dif.pvalue,4)
