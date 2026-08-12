import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any

import genesnake as gs

def benchmark(
    estimated_network: pd.DataFrame,
    reference_network: pd.DataFrame,
    plot_dir: Optional[Path] = None,
    method_name: Optional[str] = None,
    fix_pr_ylim: bool = False,
    exclude_diagonal: bool = True,
    top_k: Optional[int] = None
) -> Dict[str, Any]:
    """
    Benchmarks an estimated GRN against a boolean reference network.
    Validates structure, removes the diagonal (optional), and calculates stats.
    """

    # Differently sorted networks could be such a nasty bug,
    # so try helping unsuspecting users spot it. Perhaps inconvenient,
    # but nothing compared to getting it wrong, so it's motivated.
    conditions = []
    for axis in ('index', 'columns'):
        conditions.append(
            (getattr(estimated_network, axis) == getattr(reference_network, axis)).all()
        )
        
    if not all(conditions):
        raise ValueError(
            'It seems either index or columns differs between reference_network '
            'and estimated_network. Are they sorted the same?'
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

    # Linearise the networks (exclude self-regulating diagonal if requested)
    if exclude_diagonal:
        m = np.full(shape=estimated_network.shape, fill_value=False)
        np.fill_diagonal(m, val=True)

        def exclude_diagonal_and_linearise(array):
            return np.ma.masked_array(array, mask=m).ravel().compressed()

        estimated_network_lin = exclude_diagonal_and_linearise(estimated_network)
        reference_network_lin = exclude_diagonal_and_linearise(reference_network)
    else:
        estimated_network_lin = np.array(estimated_network).flatten()
        reference_network_lin = np.array(reference_network).flatten()

    # Convert to 1D arrays for downstream metrics
    estimated_network_1d = np.abs(estimated_network_lin)
    reference_network_1d = reference_network_lin

    # Calculate statistics
    stats: Dict[str, Any] = {}
    
    stats.update(gs.benchmarking.calculate_auroc_and_aupr(
        estimated_network=estimated_network_1d,
        reference_network=reference_network_1d,
    ))
    
    stats['ERMA'] = gs.benchmarking.calculate_ERMA(reference_network_1d)
    
    # --- NEW: Pass top_k parameter to beeline stats ---
    stats['epr'] = gs.benchmarking.calculate_beeline_stats.early_precision_rate(
        inferred_network=estimated_network_1d,
        reference_network=reference_network_1d,
        top_k=top_k
    )
    
    stats['top_k_accuracy'] = gs.benchmarking.calculate_beeline_stats.top_k_accuracy(
        inferred_network=estimated_network_1d,
        reference_network=reference_network_1d,
        top_k=top_k
    )
    # --------------------------------------------------
    
    stats['f1_scores'] = gs.benchmarking.calculate_f1_scores(
        estimated_network=estimated_network_1d,
        reference_network=reference_network_1d,
    )
    
    if plot_dir is not None:
        gs.benchmarking.plot_auroc_and_aupr(
            stats=stats,
            method_name=method_name,
            plot_dir=plot_dir,
            fix_pr_ylim=fix_pr_ylim
        )
        
    # Safely remove plot metrics without throwing KeyError if it doesn't exist
    stats.pop('plot_metrics', None)
    
    return stats
