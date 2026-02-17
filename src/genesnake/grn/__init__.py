import lazy_loader as lazy

# Define what to lazily load
submod_attrs = {
    'make_FFLATnetwork': ['make_FFLATTnetwork'],
    'make_randomnetwork': ['make_random'],
    'make_DAG': ['make_DAG'],
    'make_scalefree': ['make_scalefree'],
    'load_and_save': ['to_edgelist', 'load', 'from_networkx', 'from_numpy', 'to_networkx'],
}

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs=submod_attrs
)
