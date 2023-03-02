import numpy as np
import pandas as pd
import sys
from sklearn.linear_model import Lasso, ElasticNet
#from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from scipy.stats import shapiro, kstest
import statistics as st
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

def cutoff_handler(A, zvec, zvec_auto=False):
    """
    Create N networks under various cutoffs using vector of so-called zeta values as thresholds
    INPUT:
        A - network
        zvec - zeta vector
        zvec_auto - 100 lineary-spaced thresholds between 0 and absolute maximum value taken from the full network
    """
    aests = []
    if (zvec_auto):
        zvec = np.linspace(0, max(abs(A)), 100)
    else:
        ep = sys.float_info.epsilon
        zmin = min(abs(A[A > 0])) - ep
        zmax = max(abs(A[A > 0])) + ep * 10
        zvec = zvec * (zmax - zmin) + zmin
    for z in range(1, len(zvec)):
        Atmp = A
        Atmp[abs(Atmp) <= zvec[z]] = 0
        aests.append(round(Atmp, ndigits=5))
    return aests


def infer_networks(Y, P, method, zvec):
    """
    Function for inferring networks with various methods
    INPUT:
        Y - gene expressions, where columns are experiments and rows are genes
        P - binary matrix of perturbations corresponding to Y
        method - inference method to apply
        zvec - vector of zeta values for cutoff
    """
    if method == "lsco":  # least squares
        aest = np.matmul(-P, np.linalg.pinv(Y))
        aests = cutoff_handler(aest, zvec, False)
    if method == "lscon":  # normalized least squares
        aest = np.matmul(-P, np.linalg.pinv(Y))
        aest = pd.DataFrame(aest)
        for column in aest:
            aest[column] = np.divide(aest[column], sum(abs(aest[column]))) * Y.shape[0]
        aests = cutoff_handler(aest, zvec, False)
    if method == "zscore":  # Z-score
        outNet = []
        P2 = P.transpose()
        for i in P2:
            inds = np.nonzero(abs(P.iloc[i].array))
            Y2 = Y.transpose()
            tmpAest = np.divide((np.mean(Y2.iloc[inds], axis=0) - np.mean(Y, axis=1)), np.std(Y, axis=1))
            outNet.append(tmpAest)
        aest = pd.DataFrame(outNet).transpose()
        aests = cutoff_handler(aest, zvec, False)
    if method == "lasso":  # lasso
        aests = []
        for z in zvec:
            lasso_mod = Lasso(alpha=z)
            out_net_tmp = []
            for i in P.transpose():
                lasso_mod.fit(Y.transpose(), -P.iloc[i])
                out_net_tmp.append(lasso_mod.coef_)
            out_net_tmp = pd.DataFrame(out_net_tmp)
            aests.append(out_net_tmp)
    if method == "elnet":  # elastic net
        aests = []
        for z in zvec:
            lassoMod = ElasticNet(alpha=z, random_state=0)
            outNetTmp = []
            for i in P.transpose():
                lassoMod.fit(Y.transpose(), -P.iloc[i])
                outNetTmp.append(lassoMod.coef_)
            outNetTmp = pd.DataFrame(outNetTmp)
            aests.append(outNetTmp)
    return aests ###

def compare_networks(nets, net0, exclude_diagonal=False, signed=False):
    """
    Function for comparing networks and returning performance statistics.
    Allows to exclude diagonal and/or consider sign in a network.
    INPUT:
        nets - inferred networks
        net0 - gold standard network for comparison
    """
    TPo = []
    FPo = []
    TNo = []
    FNo = []

    TPRo = []
    TNRo = []
    PPVo = []
    NPVo = []
    FNRo = []
    FPRo = []
    FDRo = []
    FORo = []
    LRpo = []
    LRmo = []
    PTo = []
    TSo = []

    F1o = []
    MCCo = []
    FMo = []
    BMo = []
    MKo = []
    DORo = []
    So = []

    for n in range(0, len(nets)):

        if (exclude_diagonal):  # if diagonal should be excluded for statistics
            net1 = np.array(nets[n])  # take a single network to compare
            np.fill_diagonal(net1, 0)
            net2 = np.array(net0)  # gold standard
            np.fill_diagonal(net2, 0)
        else:  # if diagonal should be included for statistics
            net1 = np.array(nets[n])  # take a single network to compare
            net2 = np.array(net0)  # gold standard

        S = len(np.nonzero(net1)[0])/net1.shape[0] # sparsity of a network

        So.append(S)

        if (signed):  # if sign should be considered
            net1[net1 > 0] = 1
            net1[net1 < 0] = -1
            net2[net2 > 0] = 1
            net2[net2 < 0] = -1
            cond1 = net1 == net2
            cond2 = net1 != 0
            cond3 = net2 != 0

            TP = sum(sum(cond1 & cond2 & cond3))
            TPo.append(TP)
            FP = sum(sum(net1.astype(dtype=bool))) - TP
            FPo.append(FP)
            TN = sum(sum(~net1.astype(dtype=bool) & ~net1.astype(dtype=bool)))
            TNo.append(TN)
            FN = sum(sum(~net1.astype(dtype=bool))) - TN
            FNo.append(FN)
        else:  # if sign should be ignored
            net1 = net1.astype(dtype=bool)
            net2 = net2.astype(dtype=bool)
            TP = sum(sum(net1 & net2))
            TPo.append(TP)
            FP = sum(sum(net1)) - TP
            FPo.append(FP)
            TN = sum(sum(~net1 & ~net2))
            TNo.append(TN)
            FN = sum(sum(~net1)) - TN
            FNo.append(FN)

        #precision, recall, thresholds = precision_recall_curve((net2+[0]).flatten(), (net1+[0]).flatten())
        # Use AUC function to calculate the area under the curve of precision recall curve
        #auc_precision_recall = auc(recall, precision)
        #roc_auc = roc_auc_score((net2+[0]).flatten(), (net1+[0]).flatten())

        # AUPRo.append(auc_precision_recall)
        # AUCo.append(roc_auc)

        # measures based on confusion matrix
        # based on https://en.wikipedia.org/wiki/Confusion_matrix
        # ifs are for handling division by zero

        # sensitivity
        if (TP + FN) != 0:
            TPR = TP / (TP + FN)
        else:
            TPR = 0
        TPRo.append(TPR)
        # specificity
        if (TN + FP) != 0:
            TNR = TN / (TN + FP)
        else:
            TNR = 0
        TNRo.append(TNR)
        # precision
        if (TP + FP) != 0:
            PPV = TP / (TP + FP)
        else:
            PPV = 0
        PPVo.append(PPV)
        # negative predictive value
        if (TN + FN) != 0:
            NPV = TN / (TN + FN)
        else:
            NPV = 0
        NPVo.append(NPV)
        # miss rate
        if (FN + TP) != 0:
            FNR = FN / (FN + TP)
        else:
            FNR = 0
        FNRo.append(FNR)
        # fall-out
        if (FP + TN) != 0:
            FPR = FP / (FP + TN)
        else:
            FPR = 0
        FPRo.append(FPR)
        # false discovery rate
        if (FP + TP) != 0:
            FDR = FP / (FP + TP)
        else:
            FPR = 0
        FDRo.append(FDR)
        # false omission rate
        if (FN + TN) != 0:
            FOR = FN / (FN + TN)
        else:
            FOR = 0
        FORo.append(FOR)
        # positive likelihood ratio
        if FPR != 0:
            LRp = TPR / FPR
        else:
            LRp = 0
        LRpo.append(LRp)
        # negative likelihood ratio
        if TNR != 0:
            LRm = FNR / TNR
        else:
            LRm = 0
        LRmo.append(LRm)
        # prevalence threshold
        if (np.sqrt(TPR) + np.sqrt(FPR)) != 0:
            PT = np.sqrt(FPR) / (np.sqrt(TPR) + np.sqrt(FPR))
        else:
            PT = 0
        PTo.append(PT)
        # threat score
        if (TP + FN + FP) != 0:
            TS = TP / (TP + FN + FP)
        else:
            TS = 0
        TSo.append(TS)
        # F1 score
        if (2 * TP + FP + FN) != 0:
            F1 = 2 * TP / (2 * TP + FP + FN)
        else:
            F1 = 0
        F1o.append(F1)
        # MCC
        MCC = np.sqrt(PPV * TPR * TNR * NPV) - np.sqrt(FDR * FNR * FPR * FOR)
        MCCo.append(MCC)
        # Fowlkes-Mallows index
        FM = np.sqrt(PPV * TPR)
        FMo.append(FM)
        # informedness
        BM = TPR + TNR - 1
        BMo.append(BM)
        # markedness
        MK = PPV + NPV - 1
        MKo.append(MK)
        #diagnostic odds ratio
        if LRm != 0:
            DOR = LRp / LRm
            DORo.append(DOR)
        else:
            DORo.append(0)

    # calculating AUPR and AUROC

    srt = np.argsort(np.transpose(np.asarray(So)))
    TPRate = np.transpose(np.asarray(TPRo))
    FPRate = np.transpose(np.asarray(FPRo))
    Prec = np.transpose(np.asarray(PPVo))

    AUROCo = round(np.trapz(TPRate[srt], FPRate[srt]), ndigits=4)
    AUPRo = round(np.trapz(Prec[srt], TPRate[srt]), ndigits=4)

    data = [TPo, FPo, TNo, FNo,
        TPRo, TNRo, PPVo, NPVo, FNRo, FPRo, FDRo, FORo, LRpo, LRmo, PTo, TSo,
        F1o, MCCo, FMo, BMo, MKo, DORo, So]
    df = pd.DataFrame(data,
                    index=['tp', 'fp', 'tn', 'fn',
                         'tpr', 'tnr', 'ppv', 'npv', 'fnr', 'fpr', 'fdr', 'for', 'lrp', 'lrm','pt','ts',
                         'f1', 'mcc', 'fm', 'bm', 'mk', 'dor', 's'])
    op = dict()
    df2 = df.replace(np.nan, 0)
    op['metrics'] = df2.transpose()
    op['AUROC'] = AUROCo
    op['AUPR'] = AUPRo
    return op


def data_summary(data, p_matrix):
    """
    Function for displaying data summary.
    INPUT:
        data - Y matrix
    """

    nrw = len(data.index)
    ncl = len(data.columns)

    data2 = data.to_numpy()
    test_nor = kstest(data2.flatten(), 'norm')

    desc = list(['0 values', 'Inf values', 'NaN values', 'positive values', 'negative values'])
    vals = np.asarray([np.isclose(data, 0.0).sum(),
           np.isinf(data).sum().sum(),
           data.isnull().values.sum(),
           (data > 0).values.sum(),
           (data < 0).values.sum()])/(nrw*ncl)*100

    # perturbation ranking
    #for i in range(0, p_matrix.shape[1]-1):
    #    print(np.array(p_matrix.ix[:,i]))

    pert_rank = []

    for i, column in enumerate(p_matrix):
        ind_p = p_matrix.iloc[:,i] == 1
        tmp_data_col = data[i]
        pert = abs(np.array(tmp_data_col[np.invert(ind_p)])) > abs(np.array(tmp_data_col[ind_p]))
        #print(sum(pert == True))
        #print(type(pert))
        tmp_sum = sum(pert == True).tolist()
        #print((tmp_sum[0]))
        pert_rank.append(tmp_sum)


    # For Sine Function

    figure, axes = plt.subplots(2, 2, figsize=(12, 12))
    sns.kdeplot(np.array(data).flatten(), ax = axes[0, 0])
    axes[0, 0].title.set_text('Data density')
    sm.qqplot(data, line='45', ax = axes[0, 1])
    axes[0, 1].text(-1.5, 1.5, 'KS P value = '+str(round(test_nor.pvalue, ndigits=2)), fontsize=12)
    axes[0, 1].title.set_text('Data normality')
    axes[1, 0].bar(desc, vals, color ='slategrey', width = 0.4)
    axes[1, 0].title.set_text('Values in the data')
    axes[1, 0].set_ylabel("Amount [%]")
    axes[1, 0].set_xticklabels(desc, rotation=45)
    # Combine all the operations and display
    sns.histplot(pert_rank, ax = axes[1, 1], kde=True, element="step")
    axes[1, 1].title.set_text('Perturbation ranking')
    plt.show()

import random
import pandas as pd
import numpy as np
from pathlib import Path
import statistics as stats
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
            rate = np.divide(counts_samp, c*g_length_set/L) # estimated / expected
            snr = np.var(c*g_length_set/L)/np.var(counts_samp - c*g_length_set/L)  # var (signal) / var (noise)

            allArrays[s].append(np.transpose(rate))
            all_snrs.append(snr)
        out = np.transpose(pd.DataFrame(np.concatenate(allArrays)))  # samples in columns, genes in rows
    out += out

    op = dict()
    op['ratios'] = out/techrep
    op['snrs'] = all_snrs # snrs per sample

    return op
