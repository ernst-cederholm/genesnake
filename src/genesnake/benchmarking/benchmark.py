
from pathlib import Path
from .compare_networks import compare_networks
from .cutoff_handler import cutoff_handler
from .auroc_and_aupr import auroc_and_aupr

def benchmark(
        estimated_network,
        true_network,
        output_dir,
        model_name,
        ):

    inets = cutoff_handler(estimated_network)
    out = compare_networks(inets, true_network)
    outm = out['metrics']

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok = True)
    outm.to_csv(
        path_or_buf = output_dir / f'benchmark_stats_{model_name}.csv',
        sep = '\t', header = True, index = True)

    print(
        auroc_and_aupr(
            full_network = estimated_network,
            true_network = true_network,
            metrics = outm,
            model_name = model_name,
            output_dir = output_dir,
            fix_ylim = False)
        )
