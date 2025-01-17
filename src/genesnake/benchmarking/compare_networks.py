import numpy as np
import pandas as pd

def compare_networks(
		networks,
		true_network,
		exclude_diag = True,
		include_sign = False
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

			PP = net1.astype(dtype=bool)  # Predicted Positive
			AP = net2.astype(dtype=bool)  # Actual Positive

			FP = sum(sum(PP)) - TP
			FPo.append(FP)
			TN = sum(sum(~PP & ~AP))
			TNo.append(TN)
			FN = sum(sum(~PP)) - TN
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
		TPR = TP / (TP + FN)
		TPRo.append(TPR)
		# specificity
		TNR = TN / (TN + FP)
		TNRo.append(TNR)
		# precision
		PPV = TP / (TP + FP)
		PPVo.append(PPV)
		# negative predictive value
		NPV = TN / (TN + FN)
		NPVo.append(NPV)
		# miss rate
		FNR = FN / (FN + TP)
		FNRo.append(FNR)
		# fall-out
		FPR = FP / (FP + TN)
		FPRo.append(FPR)
		# false discovery rate
		FDR = FP / (FP + TP)
		FDRo.append(FDR)
		# false omission rate
		FOR = FN / (FN + TN)
		FORo.append(FOR)
		# positive likelihood ratio
		LRp = TPR / FPR
		LRpo.append(LRp)
		# negative likelihood ratio
		LRm = FNR / TNR
		LRmo.append(LRm)
		# prevalence threshold
		PT = np.sqrt(FPR) / (np.sqrt(TPR) + np.sqrt(FPR))
		PTo.append(PT)
		# threat score
		TS = TP / (TP + FN + FP)
		TSo.append(TS)
		# F1 score
		F1 = 2 * TP / (2 * TP + FP + FN)
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
		DOR = LRp / LRm
		DORo.append(DOR)

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
