
import numpy as np
import pandas as pd
from pathlib import Path
from .calculate_auroc_and_aupr import calculate_auroc_and_aupr
from .plot_auroc_and_aupr import plot_auroc_and_aupr
from .calculate_ERMA import calculate_ERMA
from . import calculate_beeline_stats

def benchmark(
		estimated_network,
		reference_network,
		plot_dir = None,
		method_name = None,
		fix_pr_ylim = False,
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
			'It seems either index or columns differs between reference_network' +
			' and estimated_network. Are they sorted the same?'
			)
	# Had to debug for this myself once, so added check for future //AB
	for net_name, net in zip(
			['estimated_network', 'reference_network'],
			[estimated_network, reference_network]
			):
		if np.isnan(net).any().any():
			print(f'Warning: Discovered NaN values in {net_name}. If not intentional, this might give inaccurate benchmark results')

	stats = {}
	stats = calculate_auroc_and_aupr(
		estimated_network = estimated_network,
		reference_network = reference_network,
		)
	stats['ERMA'] = calculate_ERMA(reference_network)
	stats['epr'] = calculate_beeline_stats.early_precision_rate(
		inferred_network = estimated_network,
		reference_network = reference_network,
		)
	stats['top_k_accuracy'] = calculate_beeline_stats.top_k_accuracy(
		inferred_network = estimated_network,
		reference_network = reference_network,
		)

	if plot_dir is not None:
		plot_auroc_and_aupr(
			stats = stats,
			method_name = method_name,
			plot_dir = plot_dir,
			fix_pr_ylim = False
			)

	stats.pop('plot_metrics')
	return(stats)
