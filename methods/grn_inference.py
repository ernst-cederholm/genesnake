import numpy as np
import pandas as pd
import sys
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet


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

        S = np.mean(sum(net1))  # sparsity of a network
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
        FPRo.append(FPRo)
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

    data = [TPo, FPo, TNo, FNo,
            TPRo, TNRo, PPVo, NPVo, FNRo, FPRo, FDRo, FORo, LRpo, LRmo, PTo, TSo,
            F1o, MCCo, FMo, BMo, MKo, DORo, So]
    df = pd.DataFrame(data,
                      index=['tp', 'fp', 'tn', 'fn',
                             'tpr', 'tnr', 'ppv', 'npv', 'fnr', 'fpr', 'fdr', 'for', 'lrp', 'lrm','pt','ts',
                             'f1', 'mcc', 'fm', 'bm', 'mk', 'dor', 's'])
    df2 = df.replace(np.nan, 0)
    return df2.transpose()