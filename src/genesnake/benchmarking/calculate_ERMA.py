import math
import numpy as np

def calculate_ERMA(true_network):
    """
    Calculate Exptected Random Model AUPR (ERMA)
    ERMA = P / A, 
    A = N + P, 
    P is (actual) Positives, N is (actual) Negatives
    
    The intuition is that a random model will always have precision P / A
    for any sparsity of prediction, since it guesses randomly between
    PP (Predicted Positive) and PN (Predicted Negative).
    y axis of PR plot = Precision = P / A.
    x axis length is 1.
    Area of rectangle = height x width = (P / A) * 1 = P / A
    
    Note that ERMA depends on the problem sparsity. Unlike AUROC,
    the baseline changes depending on the problem. That's why it's
    useful to include this number with the AUPR.
    
    Reference:
        Links
            https://doi.org/10.1371/journal.pone.0118432
            https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432#pone-0118432-g005
        Citation
            Saito T, Rehmsmeier M. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLoS One. 2015 Mar 4;10(3):e0118432. doi: 10.1371/journal.pone.0118432. PMID: 25738806; PMCID: PMC4349800.
    """
    A = math.prod(true_network.shape)
    P = len(np.array(true_network).nonzero()[0])
    ERMA = P / A
    return(ERMA)
