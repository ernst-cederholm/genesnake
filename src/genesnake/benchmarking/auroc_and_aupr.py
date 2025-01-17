
# This code starts from genesnake benchmark statistics and
# calculates and plots ROC and PR curves along with the areas under them


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from .compare_networks import compare_networks
import math
from .calculate_ERMA import calculate_ERMA


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
		[last_predicted_network],
		true_network,
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




def auroc_and_aupr(
	full_network, 
	true_network, 
	metrics, 
	model_name,
	output_dir,
	exclude_diag = True,
	include_sign = False,
	fix_ylim = True,
	):
	"""
	Plot ROC and PR curves, and return AUROC and AUPR

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
	Precision = Precision.bfill()
	Recall_extended = list(Recall) + list(PRe['recall'])
	Precision_extended = list(Precision) + list(PRe['precision'])
	AUPR = np.trapz(
		Precision_extended, 
		x = Recall_extended
		)

	# Calculate Exptected Random Model AUPR (ERMA)
	ERMA = calculate_ERMA(true_network)


	plt.close()
	plt.plot(
		FPRate, TPRate, 
		color = 'tab:blue', linestyle = 'solid',
		label = 'Observed')
	plt.plot(
		FPRate_extension, 
		TPRate_extension, 
		color = 'tab:blue', linestyle = 'dashed',
		label = 'Extended'
		)
	plt.xlim((0, 1))
	plt.ylim((0, 1))
	plt.xlabel('FP Rate')
	plt.ylabel('TP Rate')
	plt.legend()
	plt.text(
		0.4, 
		0.1, 
		f'AUROC (extended) = {AUROC:.3}',
		)
	title = f'{model_name}_ROC_curve'
	plt.title(title)
	plt.tight_layout()
	filename = f'{title}.svg'
	plt.savefig(output_dir / filename)

	plt.close()
	plt.plot(Recall, Precision,
		color = 'tab:blue', linestyle = 'solid',
		label = 'Observed')
	plt.plot(PRe['recall'], PRe['precision'],
		color = 'tab:blue', linestyle = 'dashed',
		label = 'Extended')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.xlim((0, 1))
	# When AUPR is really shit, one cannot see the shape of it
	# because it's just a line at the bottom. But the shape is 
	# useful to check if shape looks sane. Thus the option
	if fix_ylim:
		top = 1.1
		plt.ylim((0, top))
	else:
		ax = plt.gca()
		_, top = ax.get_ylim()
		ax.set_ylim(0, top)
	plt.legend()
	l1 = f'AUPR (DREAM2 extended) = {AUPR:.3}'
	l2 = f'ERMA = {ERMA:.3}'
	plt.text(
		0.4, top * 0.7,
		'\n'.join([l1, l2]),
		)
	title = f'{model_name}_PR_curve'
	plt.title(title)
	plt.tight_layout()
	filename = f'{title}.svg'
	plt.savefig(
		output_dir / filename)

	return({'AUROC': AUROC, 'AUPR': AUPR, 'ERMA': ERMA})


