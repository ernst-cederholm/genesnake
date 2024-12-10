
# This code starts from genesnake benchmark statistics and
# calculates and plots ROC and PR curves along with the areas under them


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import anton_util
from .compare_networks import compare_networks
import math


# The reason for splitting this rather trivial line out to its
# own function was mostly to get the extensive comment about
# the justification for the extension out of the flow of
# auroc_and_aupr for more easily read code
def extend_ROC_curve(TPRate, FPRate):
    """
    The ROC curve can be linearly extrapolated from the last 
    measured point to (1, 1). GRN Benchmark refers to DREAM5 that 
    refers to Davis and Goadrich.
    See Relevant sources in the function docstring for complete
    source names and links.
    Davis and Goadrich asserts that linear interpolation is appropriate
    for ROC, and supplies some intuition for it.
    Not sure if Davis and Goadrich mentions or shows that a ROC curve
    will always end up at (1, 1), but this follows from the definitions of
    True Positive Rate and False Positive Rate.
    TPR = TP / P, TP = True Positives, P = (Actual) Positives
    FPR = FP / N, FP = False Negatives, N = (Actual) Negatives
    Both TP and FP are functions of the number of Positive Predictions, PP.
    A positive prediction in this context is that the link exists. 
    As PP grows, so must TP and FP, and thus TPR and FPR.
    The total population (possible links) is P + N.
    The maximum PP = P + N, ie all links are predicted.
    Thus, TP and FP are the only possible outcomes. Then it must be that
    TP = P
    FP = N
    => 
    TPR = 1
    FPR = 1

    References:
        GRN Benchmark
            https://grnbenchmark.org/
            Citation
                Seçilmiş D, Hillerton T, Sonnhammer ELL. GRNbenchmark - a web server for benchmarking directed gene regulatory network inference methods. Nucleic Acids Res. 2022 Jul 5;50(W1):W398-W404. doi: 10.1093/nar/gkac377. PMID: 35609981; PMCID: PMC9252735.
        DREAM5 
            Citation
                Marbach D, Costello JC, Küffner R, Vega NM, Prill RJ, Camacho DM, Allison KR; DREAM5 Consortium; Kellis M, Collins JJ, Stolovitzky G. Wisdom of crowds for robust gene network inference. Nat Methods. 2012 Jul 15;9(8):796-804. doi: 10.1038/nmeth.2016. PMID: 22796662; PMCID: PMC3512113.
        Davis and Goadrich
            J. Davis and M. Goadrich. The relationship between
            Precision-Recall and ROC curves. In ICML ’06: Proceedings of the 23rd international conference on Machine
            learning, pages 233–240, Pittsburgh, Pennsylvania, 2006.
            ACM.
    """
    TPRate_extended, FPRate_extended  = [
        list(var) + [1] for var in (TPRate, FPRate)]
    return(TPRate_extended, FPRate_extended)




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



def calculate_AUPR_extension(
    true_network,
    last_predicted_network,
    ):
    """
    Input networks are in matrix form, not edge list. This is for compatibility with the rest of genesnake. For this specific function, edge list would probably have been more convenient.
    
    last_predicted_network is the network with the highest
    recall, ie the least stringent filtering of links.

    Returns dict: {label: array}, 
    where label is in ('recall', 'precision')

    The extension is done according to the DREAM2 challenge paper at
        link
            https://doi.org/10.1111/j.1749-6632.2009.04497.x
        Paper citation
            Stolovitzky G, Prill RJ, Califano A. Lessons from the DREAM2 Challenges. Ann N Y Acad Sci. 2009 Mar;1158:159-95. doi: 10.1111/j.1749-6632.2009.04497.x. PMID: 19348640.
    . GRN Benchmark
        https://grnbenchmark.org/
        Citation
            Seçilmiş D, Hillerton T, Sonnhammer ELL. GRNbenchmark - a web server for benchmarking directed gene regulatory network inference methods. Nucleic Acids Res. 2022 Jul 5;50(W1):W398-W404. doi: 10.1093/nar/gkac377. PMID: 35609981; PMCID: PMC9252735.
    cites DREAM5 
        Citation
            Marbach D, Costello JC, Küffner R, Vega NM, Prill RJ, Camacho DM, Allison KR; DREAM5 Consortium; Kellis M, Collins JJ, Stolovitzky G. Wisdom of crowds for robust gene network inference. Nat Methods. 2012 Jul 15;9(8):796-804. doi: 10.1038/nmeth.2016. PMID: 22796662; PMCID: PMC3512113.
    that refers back to DREAM2 for this procedure.

    The function internals use the nomenclature from the DREAM2 paper, so should be straight forward to follow given the paper. No further explanation is done here. Refer to the paper for context and explanations. Get in touch if understanding the paper or this code becomes a problem.
    """

    # Get some needed quantities
    stats = compare_networks(
        [last_predicted_network],
        true_network,
        exclude_diag=False, 
        include_sign=False
        )['metrics']

    # Problem below:
    # Code below assumes boolean networks.
    # For example P = true_network.sum().sum() to get the true positives
    # So I need options for boolean vs continuous networks here as well?
    #
    # No, these calculations should always use boolean.
    # Refer to compare_networks if needed.
    # What I might need is an option to include sign though...
    # 
    # compare_networks handles this already. Can I utilise that?
    # The only quantity that looks off is P.
    # P = TP + FN
    # I'll try that!
    # Seems to work ^^

    # Calculate the PRC extension according to DREAM2
    T = math.prod(true_network.shape)  # Number of possible edges
    # P = true_network.sum().sum()  # Number of true positives
    P = int(stats['tp'] + stats['fn'])
    TPL = int(stats['tp'])
    FPL = int(stats['fp'])
    rL = float(stats['tpr'])  # Could also calculate as TPL / P
    L = TPL + FPL  # Length of list of predictions done
    r = np.linspace(rL, 1, num = 100)
    
    try:
        rho = (P - TPL) / (T - L)
        p = rho * (P * r) / (P * (r - rL) + L * rho)
    # If T - L = 0, we don't have to extend AUPR, it has already
    # gone all the way. In this case, return empty r and p
    except ZeroDivisionError:
        r = []
        p = []

    return({'recall': r, 'precision': p})




def auroc_and_aupr(
    full_network, 
    true_network, 
    metrics, 
    model_name,
    output_dir):
    """
    Plot ROC and PR curves, and return AUROC and AUPR

    Extends ROC linearly to (1,1), and AUPR according to DREAM2. See code for more details and references.
    """

    # Ensure data is sorted on sparsity
    sparsity_order = np.argsort(metrics.s)
    outs = metrics.iloc[sparsity_order, :]

    TPRate = outs.tpr
    Recall = TPRate  # It's just different names for the same thing
    FPRate = outs.fpr
    Precision = outs.ppv

    # Extend ROC curve
    TPRate_extended, FPRate_extended = extend_ROC_curve(TPRate, FPRate)
    AUROC = np.trapz(TPRate_extended, x = FPRate_extended)

    # Extend PR curve
    AUPRe = calculate_AUPR_extension(
        true_network = true_network,
        last_predicted_network = full_network,
        )
    Recall_extended = list(Recall) + list(AUPRe['recall'])
    Precision_extended = list(Precision) + list(AUPRe['precision'])
    AUPR = np.trapz(
        Precision_extended, 
        x = Recall_extended
        )

    # Calculate Exptected Random Model AUPR (ERMA)
    ERMA = calculate_ERMA(true_network)

    # FIX: 
    # End of Recall is not the same as start of AUPRe['recall']
    # which I would expect. They align nicely, it just want to
    # verify why, and that it makes sense
    #
    # Actually, finding out why this is will be simpler
    # once I've cleaned up the code a bit, which I will
    # do next, putting it into genesnake.
    # So save this for later.
    # My head might be fresher at that point too.
    # It's friday evening after Davides grilling now,
    # and it's been a loong week.
    plt.close()
    plt.plot(FPRate, TPRate, color = 'tab:blue', linestyle = 'solid')
    plt.plot(
        # This looks a bit strange. Why would I hardcode (1,1) here
        # when I went through trouble above to extend it?
        # FIX
        [FPRate_extended[-2], 1], 
        [TPRate_extended[-2], 1], 
        color = 'tab:blue', linestyle = 'dashed'
        )
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel('FP Rate')
    plt.ylabel('TP Rate')
    plt.text(
        0.4, 
        0.1, 
        f'AUROC (extended to (1, 1)) = {round(AUROC, ndigits = 3)}', 
        )
    filename = f'benchmark_stats_{model_name}_ROC_curve.svg'
    plt.title(filename)
    plt.savefig(
        output_dir / filename)

    plt.close()
    plt.plot(Recall, Precision,
        color = 'tab:blue', linestyle = 'solid')
    plt.plot(AUPRe['recall'], AUPRe['precision'],
        color = 'tab:blue', linestyle = 'dashed')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim((0, 1))

    # Will probably need to include a version that doesn't
    # fix ylim. People are going to want to see the shape
    # even when the values are very small.
    # Or well, I wanted to, when I developed the code for
    # extending AUPR
    # FIX

    plt.ylim((0, 1))
    l1 = f'AUPR (DREAM2 extended) = {round(AUPR, ndigits = 3)}'
    l2 = f'ERMA = {round(ERMA, ndigits = 3)}'
    plt.text(
        0.4, 0.7, 
        '\n'.join([l1, l2]),
        )
    filename = f'benchmark_stats_{model_name}_PR_curve.svg'
    plt.title(filename)
    plt.savefig(
        output_dir / filename)

    return({'AUROC': AUROC, 'AUPR': AUPR, 'ERMA': ERMA})


