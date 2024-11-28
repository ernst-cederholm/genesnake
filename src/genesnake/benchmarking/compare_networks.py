import numpy as np
import pandas as pd

def compare_networks(nets, net_gs, exclude_diag=False, include_sign=False):
    """
    Function for comparing networks and returning performance statistics.
    Allows to exclude diagonal and/or consider sign in a network.
    INPUT:
        nets - inferred networks
        net_gs - gold standard network for comparison
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

        if (exclude_diag):  # if diagonal should be excluded for statistics
            net1 = np.array(nets[n])  # take a single network to compare
            np.fill_diagonal(net1, 0)
            net2 = np.array(net_gs)  # gold standard
            np.fill_diagonal(net2, 0)
        else:  # if diagonal should be included for statistics
            net1 = np.array(nets[n])  # take a single network to compare
            net2 = np.array(net_gs)  # gold standard

        S = len(np.nonzero(net1)[0])/net1.shape[0] # sparsity of a network

        So.append(S)

        if (include_sign):  # when sign is considered
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
            FDR = 0
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
