"""
A collection of functions that are used to randomly generate parameters for the modle.py script.
Note that they are not imported in to the data namespace only the model class. 
"""

import pandas as pd
import numpy as np
import itertools

class pertgen:

    def diagonal(genes,reps,sign):
        """
        Function that creates a simple diagonal perturbation matrix with either up or down regulation of gene expression. 
        INPUT:
            reps - how many times each gene should be perturbed 
            sign - either \"up\" or \"down\". determines the direction of the perturbation up for over expression and down for knockdowns. 
        OUTPUT: 
            pert_mat - a pandas dataframe of size genesXgenes*reps with a perturbation of +/-0.5 of each gene along the diagonal
        """
        
        # get the number of genes in the system 
        N = len(genes)
        
        # set a single replicate as a start 
        pert_mat=np.eye(N)
        columns = [str(g)+".1" for g in genes]
        
        # add a diagonal matrix for all other replicates 
        for i in range(reps-1):
            pert_mat = np.hstack([pert_mat,np.eye(N)])
            columns = columns + [str(g)+"."+str(i+2) for g in genes]
            
        # then set the values to+/-0.5 
        if sign=="up":
            pert_mat = pert_mat*0.5
        elif sign == "down":
            pert_mat = pert_mat*-0.5
        else:
            raise ValueError("Attempted to assign unknown sign to perturbation in genesnake.GRNmodel.get_perts. Only \"up\" or \"down\" ccan be used.")

        # clean up so that all 0s are positive, this does not matter from a math standpoint but
        # looks slightly better with 0 than -0
        pert_mat[np.nonzero(pert_mat==0)] = 0

        return pd.DataFrame(pert_mat,index=genes,columns=columns)
