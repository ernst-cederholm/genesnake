
import numpy as np
import pandas as pd
from pathlib import Path
from .compare_networks import compare_networks
from .calculate_auroc_and_aupr import calculate_auroc_and_aupr
from .plot_auroc_and_aupr import plot_auroc_and_aupr
from .calculate_cutoffs import calculate_cutoffs
from .cut_network import cut_network
from . import calculate_beeline_stats

def benchmark(
		estimated_network,
		true_network,
		output_dir,
		model_name,
		):
	"""
	Benchmark. A high level interface that doesn't expose all options.
	For more options, see code.
	"""

	# Differently sorted networks could be such a nasty bug to hunt down,
	# so try helping unsuspecting users spot it. Perhaps inconvenient,
	# but nothing compared to getting it wrong, so it's motivated.
	conditions = []
	for axis in ('index', 'columns'):
		conditions.append(
			(
				getattr(estimated_network, axis) ==
				getattr(true_network, axis)
				).all()
			)
	if not all(conditions):
		raise ValueError(
			'It seems either index or columns differs between true_network' +
			' and estimated_network. Are they sorted the same?'
			)
	for net_name, net in zip(
			['estimated_network', 'true_network'],
			[estimated_network, true_network]
			):
		if np.isnan(net).any().any():
			print(f'Warning: Discovered NaN values in {net_name}. If not intentional, this might give inaccurate benchmark results')

	cutoffs = calculate_cutoffs(estimated_network)
	# Retain a copy of the original estimated network, since
	# cut_network operates inplace to avoid repeatedly copying the network.
	from copy import deepcopy
	network = deepcopy(estimated_network)
	tn = deepcopy(true_network)
	ms = []
	for cutoff in cutoffs:
		cut_network(network, cutoff)
		# Warning:
		# compare_networks can modify both networks and true_network
		# This to avoid repeated copying of networks in the hot loop
		# of the benchmarking. Surrounding code has to be aware of
		# this and handle it.
		m = compare_networks(
			networks = [network],
			true_network = tn)
		ms.append(m)
	metrics = pd.concat(ms, axis = 0, ignore_index = True)

	output_dir = Path(output_dir)
	output_dir.mkdir(exist_ok = True, parents = True)
	metrics.to_csv(
		path_or_buf = output_dir / f'benchmark_stats_{model_name}.csv',
		sep = '\t', header = True, index = True)
	stats = calculate_auroc_and_aupr(
		full_network = estimated_network,
		true_network = true_network,
		metrics = metrics,
		)
	plot_auroc_and_aupr(
		stats = stats,
		model_name = model_name,
		output_dir = output_dir
		)
	epr = calculate_beeline_stats.early_precision_rate(
		inferred_network = estimated_network,
		true_network = true_network,
		)
	top_k_accuracy = calculate_beeline_stats.top_k_accuracy(
		inferred_network = estimated_network,
		true_network = true_network,
		)

	stats.pop('plot_metrics')
	return({**stats, 'EPR': epr, 'top_K_acc': top_k_accuracy})
