import importlib

__all__ = [
    "compare_networks", "data_summary", "calculate_auroc_and_aupr", 
    "plot_auroc_and_aupr", "benchmark", "calculate_ERMA", "calculate_beeline_stats"
]

# Map attribute names to their source modules
_LAZY_IMPORTS = {
    "compare_networks": "compare_networks",
    "data_summary": "data_summary",
    "calculate_auroc_and_aupr": "calculate_auroc_and_aupr",
    "plot_auroc_and_aupr": "plot_auroc_and_aupr",
    "benchmark": "benchmark",
    "calculate_ERMA": "calculate_ERMA",
}

# Submodules that should be imported as modules, not attributes
_SUBMODULES = {"calculate_beeline_stats"}

def __dir__():
    return __all__

def __getattr__(name):
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(f".{_LAZY_IMPORTS[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
