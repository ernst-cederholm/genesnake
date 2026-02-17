import lazy_loader as lazy

# Define what to lazily load
submodules = [
    'calculate_beeline_stats',
    'data_summary'
]

submod_attrs = {
    'compare_networks': ['compare_networks'],
    'calculate_auroc_and_aupr': ['calculate_auroc_and_aupr'],
    'plot_auroc_and_aupr': ['plot_auroc_and_aupr'],
    'benchmark': ['benchmark'],
    'calculate_ERMA': ['calculate_ERMA'],
    'calculate_f1_scores': ['calculate_f1_scores'],
}

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=submodules,
    submod_attrs=submod_attrs
)
