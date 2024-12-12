
def cut_network(network, cutoff):
    """
    In network, set all values < cutoff to 0

    network is a matrix ie 2D array like.
    Either numpy or pandas probably works.
    """
    network[abs(network) <= cutoff] = 0
