import lazy_loader as lazy

# Define what to lazily load
submod_attrs = {
    'conversions': ['edgelist_to_matrix', 'matrix_to_edgelist', 'harmonise_networks'],
}

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs=submod_attrs
)
