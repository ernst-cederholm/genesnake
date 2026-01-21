import importlib

__all__ = [
    "make_FFLATTnetwork", "make_random", "make_DAG", "make_scalefree",
    "to_edgelist", "load", "from_networkx", "from_numpy", "to_networkx"
]

# Map attribute names to their source modules
_LAZY_IMPORTS = {
    "make_FFLATTnetwork": "make_FFLATnetwork",
    "make_random": "make_randomnetwork",
    "make_DAG": "make_DAG",
    "make_scalefree": "make_scalefree",
    "to_edgelist": "load_and_save",
    "load": "load_and_save",
    "from_networkx": "load_and_save",
    "from_numpy": "load_and_save",
    "to_networkx": "load_and_save",
}

def __dir__():
    return __all__

def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(f".{_LAZY_IMPORTS[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
