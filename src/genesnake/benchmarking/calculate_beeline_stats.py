
import numpy as np
import genesnake as gs
import copy


def early_precision_rate(
		inferred_network, 
		reference_network,
		fraction_of_K = 1):
	"""
	Calculate Early Precision Rate (EPR) according to
	dspin bioarxiv preprint at
	doi
		https://doi.org/10.1101/2023.04.19.537364
	link (more precise, to full text version 4)
		https://www.biorxiv.org/content/10.1101/2023.04.19.537364v4.full

	Given reading a few lines of the dspin bioarxiv, I think the code
	with comments below will be understandable, so leaving it at that.
	Get in touch if problems though! You not understanding means I 
	haven't documented well enough.  //Anton Björk
	"""

	top_k_accuracy_ = top_k_accuracy(
		inferred_network, 
		reference_network,
		fraction_of_K = fraction_of_K)
	# Using ERMA to calculate the precision of a random model since
	# they are the same. See calculate_ERMA documentation for explanation
	# if curious.
	ERMA = gs.benchmarking.calculate_ERMA(reference_network)
	if ERMA == 0:
		raise ValueError(
                'ERMA is 0, which implies that the reference network has density 0. Something has gone wrong')
	else:
		early_precision_rate = top_k_accuracy_ / ERMA
	return(early_precision_rate)


def top_k_accuracy(
		inferred_network, 
		reference_network,
		fraction_of_K = 1):
	"""
	Calculate Top K accuracy according to
	dspin bioarxiv preprint at
	doi
		https://doi.org/10.1101/2023.04.19.537364
	link (more precise, to full text version 4)
		https://www.biorxiv.org/content/10.1101/2023.04.19.537364v4.full

	Given reading a few lines of the dspin bioarxiv, I think this code
	will be understandable, so leaving it at that.
	Get in touch if problems though! You not understanding means I 
	haven't documented well enough.  //Anton Björk
	"""
	K = round(len(np.array(reference_network).nonzero()[0]) * fraction_of_K)
	abs_net = np.abs(np.array(inferred_network))
	cutoff = sorted(abs_net.flatten(), reverse = True)[K - 1]
	top_K_network = copy.deepcopy(inferred_network)
	top_K_network[abs_net < cutoff] = 0

	net1 = top_K_network.astype(dtype=bool)
	# Should already be bool, but eh
	net2 = reference_network.astype(dtype=bool)

	TP = (net1 & net2).sum()
	FP = net1.sum() - TP
	TN = (~net1 & ~net2).sum()
	FN = (~net1).sum() - TN

	# precision
	if (TP + FP) == 0:
		precision = np.nan
	else:
		precision = TP / (TP + FP)

	return(float(precision))
