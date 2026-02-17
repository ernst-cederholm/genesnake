import lazy_loader as lazy

# Define what to lazily load
submodules = ["grn", "data", "util", "benchmarking", "analysis", "inference"]

submod_attrs = {
    'data.GRNmodel': ['GRNmodel'],
}

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=submodules,
    submod_attrs=submod_attrs
)
