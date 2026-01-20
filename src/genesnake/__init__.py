import importlib

__all__ = ["GRNmodel", "grn", "data", "util", "benchmarking", "analysis", "inference"]

# Map class names to (module_path, class_name) tuples
_LAZY_CLASSES = {
    "GRNmodel": ("data.GRNmodel", "GRNmodel"),
}

# Subpackages
_SUBPACKAGES = {"grn", "data", "util", "benchmarking", "analysis", "inference"}

def __dir__():
    return __all__

def __getattr__(name):
    if name in _LAZY_CLASSES:
        module_path, class_name = _LAZY_CLASSES[name]
        module = importlib.import_module(f".{module_path}", __name__)
        return getattr(module, class_name)
    if name in _SUBPACKAGES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
