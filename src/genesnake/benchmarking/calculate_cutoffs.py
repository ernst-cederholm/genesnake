
import numpy as np
def calculate_cutoffs(
		network,
		num_cutoffs = 200,
		all_cutoffs = False,
		manual_cutoffs = False,
		relative_cutoffs = None,
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
			Note that all conflicting options override num_cutoffs.
		all_cutoffs
			Return one cutoff for every edge in the network.
			Ensures the most accurate benchmarking, but may
			be computationally costly.
		manual_cutoffs
			Supply manual relative cutoffs in relative_cutoffs
		relative_cutoffs
			if manual_cutoffs = True,
			relative_cutoffs are scaled with maximum and minimum
			edge values to yield output cutoffs

	More on cutoff selection:
	Pick cutoffs from the sorted list of all network edge amplitudes
	at index wise regularly spaced intervals. The sorted list
	can be seen as the empirical cumulative distribution function
	of edge amplitudes. With cutoffs adapted to the distribution
	of the edge amplitudes, there is a good chance they
	will lead to an informative range of networks.
	"""
	if all_cutoffs and manual_cutoffs:
		raise ValueError(
			'all_cutoffs and manual_cutoffs are mutually exclusive')
	if manual_cutoffs and relative_cutoffs is None:
		raise ValueError(
			'If using manual_cutoffs you need to specify relative_cutoffs')

	absA = np.abs(network)
	if manual_cutoffs:
		mmin = absA.min().min()
		mmax = absA.max().max()
		cutoffs = relative_cutoffs * (mmax - mmin) + mmin
	else:
		sorted_elements = np.array(sorted(np.array(absA).flatten()))
		sorted_elements = sorted_elements[sorted_elements.nonzero()[0]]
		if all_cutoffs:
			cutoffs = sorted_elements
		else:
			cutoffs = sorted_elements[
				np.floor(
					np.linspace(0, len(sorted_elements) - 1, num_cutoffs)
					).astype(int)
				]
	# Add a cutoff that is smaller than 0, so that at least
	# one full network is returned.
	# Similarly, add a cutoffs that is guaranteed to lead to an empty network.
	# Covering the whole range of edge amplitudes ensures that
	# ROC and PR curves are complete.
	cutoffs = [-1] + list(cutoffs) + [max(cutoffs) + 1]

	return(cutoffs)
