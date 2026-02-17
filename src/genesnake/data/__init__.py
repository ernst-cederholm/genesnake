import lazy_loader as lazy

# Define what to lazily load
submod_attrs = {
    'GRNmodel': ['GRNmodel'],
    'make_new_ODEmodel': ['make_new_ODEmodel'],
    'generate_sequencing_noise': ['generate_sequencing_noise'],
    'noise_generators': ['noise_gen'],
    'parameter_generators': ['pargen'],
    'perturbation_generators': ['pertgen'],
    'simulators': ['simulators'],
}

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs=submod_attrs
)
