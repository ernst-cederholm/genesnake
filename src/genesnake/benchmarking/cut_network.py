
from copy import deepcopy

def cut_network(network, cutoff):
    """
    Return a copy of network where all values < cutoff are 0

    network is a matrix ie 2D array like.
    Either numpy or pandas probably works.
    """
    out = deepcopy(network)  # Copying here to avoid modifying inputs
    out[abs(out) <= cutoff] = 0

    return(out)
