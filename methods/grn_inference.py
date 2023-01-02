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
    F1o = []
    MCCo = []
    FMo = []
    TPo = []
    FPo = []
    TNo = []
    FNo = []
    BMo = []
    DORo = []
    MKo = []
    So = []
    PTo = []
    TSo = []
    PPVo = []
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

        TPR = TP / (TP + FN)
        TNR = TN / (TN + FP)
        PPV = TP / (TP + FP)
        PPVo.append(PPV)
        NPV = TN / (TN + FN)
        FNR = FN / (FN + TP)
        FPR = FP / (FP + TN)
        FDR = FP / (FP + TP)
        FOR = FN / (FN + TN)
        LRp = TPR / FPR
        LRm = FNR / TNR
        PT = np.sqrt(FPR) / (np.sqrt(TPR) + np.sqrt(FPR))
        PTo.append(PT)
        TS = TP / (TP + FN + FP)
        TSo.append(TS)
        ##
        F1 = 2 * TP / (2 * TP + FP + FN)
        F1o.append(F1)
        MCC = np.sqrt(PPV * TPR * TNR * NPV) - np.sqrt(FDR * FNR * FPR * FOR)
        MCCo.append(MCC)
        FM = np.sqrt(PPV * TPR)
        FMo.append(FM)
        BM = TPR + TNR - 1
        BMo.append(BM)
        MK = PPV + NPV - 1
        MKo.append(MK)
        DOR = LRp / LRm
        DORo.append(DOR)
    data = [TPo, FPo, TNo, FNo, MCCo, F1o, PPVo, FMo, BMo, MKo, DORo, PTo, TSo, So]
    df = pd.DataFrame(data,
                      index=['tp', 'fp', 'tn', 'fn', 'mcc', 'f1_score', 'ppv', 'fm', 'bm', 'mk', 'dor', 'pt', 'ts',
                             's'])
    df2 = df.replace(np.nan, 0)
    return df2.transpose()