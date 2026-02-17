import numpy as np
import pandas as pd

def compare_networks(
		networks,
		reference_network,
		exclude_diag = True,
		include_sign = False
		):
	"""
	Function for comparing networks and returning performance statistics.
	Allows to exclude diagonal and/or consider sign in a network.

	WARNING:
		This function modifies the networks.
		Normally I wouldn't write it as such, but since this
		is on the hot path of the benchmarking, I'll let it be.
		It bit me once so far. If it keeps happening I might change it.

	networks
		Iterable of matrix likes.
		Can be length 1.
	reference_network
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
			net2 = np.array(reference_network)
			m = np.full(shape = net1.shape, fill_value = False)
			np.fill_diagonal(m, val = True)
			net1 = np.ma.masked_array(net1, mask = m)
			net2 = np.ma.masked_array(net2, mask = m)

		else:
			net1 = np.array(networks[n])
			net2 = np.array(reference_network)

		S = len(np.nonzero(net1)[0])/net1.shape[0]
		So.append(S)

		# TLDR version of signed benchmark problems;
		# When signed, it's no longer binary classification, it's
		# [-1, 0, 1], which kills various assumptions around ROC
		# curves, eg random being 0.5, and TPR 1 always being reached.
		if (include_sign):
			raise(
				NotImplementedError(
"""
Signed benchmarking disabled until we figure out how to do it properly.
See comments for more details.
"""
					)
				)
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
			# Compensating for setting the diagonals to 0 to exclude them.
			# Note that this will need
			# restructuring when I enable signed benchmarking again,
			# The code for unsigned benchmarking has changed,
			# so any code sitting here should be considered dead
			# and up for remaking from scratch.
			# Which is what we'll probably end up doing, since
			# treating it as
			# trinary classification -> binarisation -> averaging ROCs
			# likely is a fundamentally sounder approach that will lead
			# to fundamentally different code here. So no reason
			# to test it and keep it up to date awaiting a fix.
			TN = sum(sum(~PP & ~AP)) - len(np.diag(net1))
			TNo.append(TN)
			FN = sum(sum(~PP)) - len(np.diag(net1)) - TN
			FNo.append(FN)
		else:
			net1 = net1.astype(dtype=bool)
			net2 = net2.astype(dtype=bool)

			TP = (net1 & net2).sum()
			TPo.append(TP)
			FP = net1.sum() - TP
			FPo.append(FP)
			TN = (~net1 & ~net2).sum()
			TNo.append(TN)
			FN = (~net1).sum() - TN
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
		if (TN + FN) == 0:
			NPV = np.inf
		else:
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
		if (FN + TN) == 0:
			FOR = np.inf
		else:
			FOR = FN / (FN + TN)
		FORo.append(FOR)
		# positive likelihood ratio
		LRp = TPR / FPR
		LRpo.append(LRp)
		# negative likelihood ratio
		if TNR == 0:
			LRm = np.inf
		else:
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
		# Suppress warnings for operations involving infinity values from
		# divide-by-zero cases above. When any component statistic is
		# undefined (infinite), MCC becomes mathematically undefined (NaN),
		# which is the correct result. The context manager prevents noisy
		# warnings while preserving correctness.
		with np.errstate(invalid='ignore'):
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
		if LRm == 0:
			DOR = np.inf
		else:
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
