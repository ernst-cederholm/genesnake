
import numpy as np
import genesnake as gs

def benchmark(
		estimated_network,
		reference_network,
		method_name = None,
		fix_pr_ylim = False,
		exclude_diagonal = True,
		return_plots = True,
		):

	# Differently sorted networks could be such a nasty bug,
	# so try helping unsuspecting users spot it. Perhaps inconvenient,
	# but nothing compared to getting it wrong, so it's motivated.
	conditions = []
	for axis in ('index', 'columns'):
		conditions.append(
			(
				getattr(estimated_network, axis) ==
				getattr(reference_network, axis)
				).all()
			)
	if not all(conditions):
		raise ValueError(
			'It seems either index or columns differs between'
			' reference_network'
			' and estimated_network. Are they sorted the same?'
			)
	# Had to debug for this myself once, so added check for future //AB
	for net_name, net in zip(
			['estimated_network', 'reference_network'],
			[estimated_network, reference_network]
			):
		if np.isnan(net).any().any():
			print(f'Warning: Discovered NaN values in {net_name}. If not intentional, this might give inaccurate benchmark results')
	if not all(reference_network.dtypes == bool):
		raise ValueError('The reference network should be boolean')


	if exclude_diagonal:
		m = np.full(shape = estimated_network.shape, fill_value = False)
		np.fill_diagonal(m, val = True)
		def exclude_diagonal_and_linearise(array):
			return(np.ma.masked_array(array, mask = m).ravel().compressed())
		estimated_network_lin = exclude_diagonal_and_linearise(
			estimated_network)
		reference_network_lin = exclude_diagonal_and_linearise(
			reference_network)
	else:
		estimated_network_lin = np.array(estimated_network).flatten()
		reference_network_lin = np.array(reference_network).flatten()

	estimated_network = np.abs(estimated_network_lin)
	reference_network = reference_network_lin



	stats = {}
	stats = gs.benchmarking.calculate_auroc_and_aupr(
		estimated_network = estimated_network,
		reference_network = reference_network,
		)
	stats['ERMA'] = gs.benchmarking.calculate_ERMA(reference_network)
	stats['epr'] = gs.benchmarking.calculate_beeline_stats.early_precision_rate(
		inferred_network = estimated_network,
		reference_network = reference_network,
		)
	stats['top_k_accuracy'] = gs.benchmarking.calculate_beeline_stats.top_k_accuracy(
		inferred_network = estimated_network,
		reference_network = reference_network,
		)
	stats['precision_at_k'] = gs.benchmarking.calculate_beeline_stats.precision_at_k(
		inferred_network = estimated_network,
		reference_network = reference_network,
		)
	stats['f1_scores'] = gs.benchmarking.calculate_f1_scores(
		estimated_network = estimated_network,
		reference_network = reference_network,
		)
	if return_plots:
		plot_roc, plot_pr = gs.benchmarking.plot_auroc_and_aupr(
			stats = stats,
			method_name = method_name,
			fix_pr_ylim = fix_pr_ylim
			)
		stats.pop('plot_metrics')
		stats['plot_roc'] = plot_roc
		stats['plot_pr'] = plot_pr
	return(stats)






