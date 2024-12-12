
import pandas as pd
from pathlib import Path
from .compare_networks import compare_networks
from .auroc_and_aupr import auroc_and_aupr
from .calculate_cutoffs import calculate_cutoffs
from .cut_network import cut_network


def benchmark(
        estimated_network,
        true_network,
        output_dir,
        model_name,
        ):
    cutoffs = calculate_cutoffs(estimated_network)
    ms = []
    for cutoff in cutoffs:
        network = cut_network(estimated_network, cutoff)
        m = compare_networks([network], true_network)
        ms.append(m)
    metrics = pd.concat(ms, axis = 0, ignore_index = True)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok = True)
    metrics.to_csv(
        path_or_buf = output_dir / f'benchmark_stats_{model_name}.csv',
        sep = '\t', header = True, index = True)
    print(
        auroc_and_aupr(
            full_network = estimated_network,
            true_network = true_network,
            metrics = metrics,
            model_name = model_name,
            output_dir = output_dir,
            fix_ylim = False)
        )
