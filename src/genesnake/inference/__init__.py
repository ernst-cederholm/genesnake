import importlib

__all__ = ["infer_networks"]

# Map attribute names to their source modules
_LAZY_IMPORTS = {
    "infer_networks": "grn_inference",
}

def __dir__():
    return __all__

def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(f".{_LAZY_IMPORTS[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
