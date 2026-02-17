
# This code starts from genesnake benchmark statistics and
# calculates and plots ROC and PR curves along with the areas under them

import numpy as np
import pandas as pd
import math
import copy
import genesnake as gs


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
	tpr_extension, fpr_extension  = [
		[var, 1] for var in (TPRate_end, FPRate_end)]
	return(tpr_extension, fpr_extension)




def calculate_PR_extension(
	reference_network,
	estimated_network,
	):
	"""
	Input networks are in matrix form, not edge list. This is for compatibility with the rest of genesnake. For this specific function, edge list would probably have been more convenient.
	
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
	stats = gs.benchmarking.compare_networks(
		# Copies are necessary to avoid unexpectedly modifying
		# the input networks. compare_networks can modify it's inputs
		# for performance reasons (that are not relevant here since
		# this is not on the hot path of the benchmarking).
		[copy.deepcopy(estimated_network)],
		copy.deepcopy(reference_network),
		# diagonal is already excluded. Normally this function
		# would take a square matrix, but now it's a 1D array
		# from flattening the networks without diagonals
		exclude_diag = False,
		)

	# Calculate the PRC extension according to DREAM2
	T = math.prod(reference_network.shape)  # Number of possible edges
	P = int((stats['tp'] + stats['fn']).iloc[0])
	TPL = int(stats['tp'].iloc[0])
	FPL = int(stats['fp'].iloc[0])
	rL = float(stats['tpr'].iloc[0])  # Could also calculate as TPL / P
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

	return(r_, p)




def calculate_auroc_and_aupr(
	estimated_network,
	reference_network,
	):
	"""
	Extends ROC linearly to (1,1), and AUPR according to DREAM2. See code for more details and references.
	"""
	import sklearn.metrics

	stats = {}

	tmp = sklearn.metrics.roc_curve(
		y_true = reference_network,
		y_score = estimated_network,
		)
	# Remove the extension to (1, 1) that roc_curve adds
	fpr, tpr, _ = [var[:-1] for var in tmp]
	# Get it again. See function docstring if curious. Also nice
	# to have the extension bit in separate variables for plotting later.
	tpr_extension, fpr_extension = calculate_ROC_extension(tpr[-1], fpr[-1])
	stats['AUROC'] = np.trapz(
		np.concatenate((tpr, tpr_extension)),
		x = np.concatenate((fpr, fpr_extension))
		)



	pr_curve = sklearn.metrics.precision_recall_curve(
		y_true = reference_network,
		y_score = estimated_network,
		# Lighter plots. Sounds like good idea.
		# Default for ROC curve but not here.
		drop_intermediate = True,
		)
	# Rest of code expects these arrays the other way around,
	# ie from low to high recall.
	# precision_recall_curve makes a linear extension to recall = 1,
	# which is inappropriate for PR curves.
	# See calculate_PR_extension for more details.
	# Thus, remove this extension which is stored in the last element.
	precision, recall, thresholds = [
		np.flip(var)[:-1] for var in pr_curve]
	if len(precision) == 1:
		print(
			"Warning: All elements of the estimated network excluding "
			"diagonal seems to be equal. If this is surprising it might "
			"indicate a bug."
			)
	# Precision is undefined for an empty network / set of predictions,
	# which happens at the low density extreme of a PR curve. Seems that
	# precision_recall_curve sets this undefined precision to 1. This is
	# annoying when AUPR is very low, because it displays the y axis up to the
	# maximum data point which makes most of the curve hard to distinguish.
	# Instead, let the first precision value be the same as the first well
	# defined precision value. This doesn't screw up the plots, and should also
	# represent a better guess/impute than 1, which is often improbable,
	# possibly inflating the AUPR in some fairly likely cases. At this point of
	# the code, precision can be length 1, so cannot do
	# precision[0] = precision[1]
	# without triggering errors in some (valid, but corner) cases. Thus simply
	# invalidate it for now and backfill later.
	precision[0] = np.nan

	# The last point added on by precision_recall_curve is
	# the correct precision endpoint, so should be able to just
	# feed that in here instead of the networks, and not have to
	# call compare networks inside this function for that precision value.
	#
	# Edit; Well, that might be true, but the function uses
	# several metrics from compare_networks that I don't have
	# on hand here, so it makes little sense to substitute it if it works.
	recall_extension, precision_extension = calculate_PR_extension(
		reference_network = reference_network,
		estimated_network = estimated_network,
		)

	all_recall = np.concatenate((recall, recall_extension))
	# Precision for an empty network is undefined, which
	# sometimes lead to the first elements of precision
	# being nan. Backfill those values with the first valid
	# precision value.
	all_precision = np.array(
		pd.Series(
			np.concatenate((precision, precision_extension))
			).bfill()
		)

	stats['AUPR'] = np.trapz(
		all_precision,
		x = all_recall
		)

	return(
		{
			**stats,
			'plot_metrics': {
				'tpr': tpr,
				'tpr_extension': tpr_extension,
				'fpr': fpr,
				'fpr_extension': fpr_extension,
				'recall': recall,
				'recall_extension': recall_extension,
				'precision': precision,
				'precision_extension': precision_extension,
				}
			}
		)
