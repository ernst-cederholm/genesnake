import numpy as np
import pandas as pd

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
            print(TP)
            FP = sum(sum(net1.astype(dtype=bool))) - TP
            FPo.append(FP)
            print(FP)

            TN = sum(sum(~net1.astype(dtype=bool) & ~net1.astype(dtype=bool)))
            TNo.append(TN)
            print(TN)
            FN = sum(sum(~net1.astype(dtype=bool))) - TN
            FNo.append(FN)
            print(FN)
        else:  # if sign should be ignored
            net1 = net1.astype(dtype=bool)
            net2 = net2.astype(dtype=bool)
            TP = sum(sum(net1 & net2))
            TPo.append(TP)
            print(TP)
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