
# This code starts from genesnake benchmark statistics and
# calculates and plots ROC and PR curves along with the areas under them

import numpy as np
from .compare_networks import compare_networks
import math
from .calculate_ERMA import calculate_ERMA
import copy


# The reason for splitting this rather trivial line out to its
# own function was mostly to get the extensive comment about
# the justification for the extension out of the flow of
# auroc_and_aupr for more easily read code
def calculate_ROC_extension(TPRate_end, FPRate_end):
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
	TPRate_extension, FPRate_extension  = [
		[var, 1] for var in (TPRate_end, FPRate_end)]
	return(TPRate_extension, FPRate_extension)




def calculate_PR_extension(
	true_network,
	last_predicted_network,
	exclude_diag,
	include_sign,
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
		# Copies are necessary to avoid unexpectedly modifying
		# the input networks. compare_networks can modify it's inputs
		# for performance reasons (that are not relevant here since
		# this is not on the hot path of the benchmarking).
		[copy.deepcopy(last_predicted_network)],
		copy.deepcopy(true_network),
		exclude_diag = exclude_diag,
		include_sign = include_sign
		)


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
	# The underscore is to avoid collision with debugger return command
	r_ = np.linspace(rL, 1, num = 100)

	# If any of these are true, the AUPR has already
	# gone all the way and we don't need to extend it
	if T - L == 0 or all(r_ - rL == 0):
		r_ = []
		p = []
	else:
		rho = (P - TPL) / (T - L)
		p = rho * (P * r_) / (P * (r_ - rL) + L * rho)

	return({'recall': r_, 'precision': p})




def calculate_auroc_and_aupr(
	full_network, 
	true_network, 
	metrics, 
	exclude_diag = True,
	include_sign = False,
	):
	"""
	Return AUROC and AUPR

	Extends ROC linearly to (1,1), and AUPR according to DREAM2. See code for more details and references.
	"""

	sparsity_order = np.argsort(metrics.s)
	outs = metrics.iloc[sparsity_order, :]

	TPRate = outs.tpr
	Recall = TPRate  # It's just different names for the same thing
	FPRate = outs.fpr
	Precision = outs.ppv

	TPRate_extension, FPRate_extension = calculate_ROC_extension(
		TPRate.iloc[-1], 
		FPRate.iloc[-1])
	AUROC = np.trapz(
		list(TPRate) + list(TPRate_extension), 
		x = list(FPRate) + list(FPRate_extension)
		)

	PRe = calculate_PR_extension(
		true_network = true_network,
		last_predicted_network = full_network,
		exclude_diag = exclude_diag,
		include_sign = include_sign,
		)
	Recall_extension = PRe['recall']
	Precision_extension = PRe['precision']
	Precision = Precision.bfill()
	AUPR = np.trapz(
		list(Precision) + list(Precision_extension),
		x = list(Recall) + list(Recall_extension)
		)

	# Calculate Exptected Random Model AUPR (ERMA)
	ERMA = calculate_ERMA(true_network)

	return(
		{
			# The print format of AUROC and AUPR was annoying, specifying
			# they were np.float64(the actual number)
			'AUROC': float(AUROC),
			'AUPR': float(AUPR),
			'ERMA': ERMA,
			'plot_metrics': {
				'TPRate': TPRate,
				'TPRate_extension': TPRate_extension,
				'FPRate': FPRate,
				'FPRate_extension': FPRate_extension,
				'Recall': Recall,
				'Recall_extension': Recall_extension,
				'Precision': Precision,
				'Precision_extension': Precision_extension,
				}
			}
		)


