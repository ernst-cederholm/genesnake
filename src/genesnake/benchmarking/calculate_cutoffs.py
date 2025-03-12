
import numpy as np
def calculate_cutoffs(
		network,
		num_cutoffs = 200,
		all_cutoffs = False,
		):
	"""
	Calculate suitable cutoffs for investigating a range of network sparsities.

	Pick representative cutoffs from the networks edge amplitude distribution.
	After picking the other cutoffs, includes ones such that at least
	one unmodified network and one full network are included in
	the output list of networks. This to ensure as complete ROC and PR
	curves as possible.

	Inputs
		network
			A network in matrix form
		num_cutoffs
			The number of cutoffs to generate.
			A higher number will give more accurate benchmarking,
			but requires more computation.
		all_cutoffs
			Return one cutoff for every edge in the network.
			Ensures the most accurate benchmarking, but may
			be computationally costly.

	More on cutoff selection:
	Pick cutoffs from the sorted list of all network edge amplitudes.
	The sorted list can be seen as the empirical cumulative distribution
	function of edge amplitudes, so it ensures they are representative.
	From it, two sets are picked, at index wise linearly spaced and
	logarithmically spaced intervals respectively. These two sets are
	then merged. The logarithmically spaced set is added since
	these indices are partcularly useful for high resolution early on
	PR curves, where they tend to vary dramatically. With early, I mean at low sparsities.
	"""

	absA = np.abs(network)
	sorted_elements = np.array(sorted(np.array(absA).flatten()))
	sorted_elements = sorted_elements[sorted_elements.nonzero()[0]]
	if all_cutoffs:
		cutoffs = sorted_elements
	else:
		cutoffs_linspace = sorted_elements[
			np.floor(
				np.linspace(
					start = 0,
					stop = len(sorted_elements) - 1,
					num = round(num_cutoffs / 2))
				).astype(int)
			]
		logspace_indices = np.unique(np.logspace(
			start = np.log10(1),
			stop = np.log10(len(sorted_elements) - 1),
			num = round(num_cutoffs / 2)).astype(int))
		cutoffs_logspace = np.flip(np.flip(sorted_elements)[logspace_indices])
		cutoffs = np.concatenate((cutoffs_linspace, cutoffs_logspace))

	# Add a cutoff that is smaller than 0, so that at least
	# one full network is returned.
	# Similarly, add a cutoffs that is guaranteed to lead to an empty network.
	# Covering the whole range of edge amplitudes ensures that
	# ROC and PR curves are complete.
	cutoffs = np.sort(np.concatenate(
		(
			[-1],
			cutoffs,
			[max(cutoffs) + 1]
			)
		))

	return(cutoffs)
