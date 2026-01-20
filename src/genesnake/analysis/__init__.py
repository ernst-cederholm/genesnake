import importlib

__all__ = ["snakeda", "umap", "network_data_dyn_test"]

# Map attribute names to (module_path, attribute_name) tuples
_LAZY_IMPORTS = {
    "snakeda": ("snakeda.snakeda", "snakeda"),
    "umap": ("umap_project_and_plot", "umap"),
    "network_data_dyn_test": ("network_to_data_dynamics_test", "network_data_dyn_test"),
}

def __dir__():
    return __all__

def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(f".{module_path}", __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
