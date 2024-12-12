import numpy as np
import pandas as pd

def compare_networks(
		networks, true_network,
		exclude_diag=False, include_sign=False
		):
	"""
	Function for comparing networks and returning performance statistics.
	Allows to exclude diagonal and/or consider sign in a network.

	networks
		Iterable of matrix likes.
		Can be length 1 to compare
	true_network
		matrix like
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

	for n in range(0, len(networks)):

		if (exclude_diag):
			net1 = np.array(networks[n])
			np.fill_diagonal(net1, 0)
			net2 = np.array(true_network)
			np.fill_diagonal(net2, 0)
		else:
			net1 = np.array(networks[n])
			net2 = np.array(true_network)

		S = len(np.nonzero(net1)[0])/net1.shape[0]

		So.append(S)

		if (include_sign):
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
		else:
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

		### Measures of confusion matrix based on
		# https://en.wikipedia.org/wiki/Confusion_matrix
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
			PPV = np.nan
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

	data = [
		TPo, FPo, TNo, FNo,
		TPRo, TNRo, PPVo, NPVo,
		FNRo, FPRo, FDRo, FORo,
		LRpo, LRmo, PTo, TSo,
		F1o, MCCo, FMo, BMo,
		MKo, DORo, So
		]
	df = pd.DataFrame(
		data,
		index=[
			'tp', 'fp', 'tn', 'fn',
			'tpr', 'tnr', 'ppv', 'npv',
			'fnr', 'fpr', 'fdr', 'for',
			'lrp', 'lrm','pt','ts',
			'f1', 'mcc', 'fm', 'bm',
			'mk', 'dor', 's'
			]
		)

	metrics = df.transpose()
	return(metrics)
