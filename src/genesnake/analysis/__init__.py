import lazy_loader as lazy

# Define what to lazily load
submod_attrs = {
    'snakeda.snakeda': ['snakeda'],
    'umap_project_and_plot': ['umap'],
    'network_to_data_dynamics_test': ['network_data_dyn_test'],
}

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs=submod_attrs
)
