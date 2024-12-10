
import numpy as np
from copy import deepcopy
def cutoff_handler(A, zvec = np.logspace(-5, 0, 100), zvec_auto=False):
    """
    Create N networks under various cutoffs using vector of so-called zeta values as thresholds. 
    Adds a zeta < 0, to ensure that at least one
    unmodified network is made.
    INPUT:
        A - a full network
        zvec - zeta vector, e.g. np.logspace(-5, 0, 100)
        zvec_auto - 100 lineary-spaced thresholds between 0 and absolute maximum value taken from the full network
    """
    absA = np.abs(A)
    aests = []
    if (zvec_auto):
        sorted_elements = np.array(sorted(np.array(absA).flatten()))
        zvec = sorted_elements[
            np.floor(
                np.linspace(0, len(sorted_elements) - 1, 100)
                ).astype(int)
            ]
    else:
        zmin = absA.min().min()
        zmax = absA.max().max()
        zvec = zvec * (zmax - zmin) + zmin

    Atmp = deepcopy(A)
    # Add a cutoff that is smaller than 0, so that at least
    # one full network is returned.
    # Similarly, add a zvec that is guaranteed to lead to an empty network.
    # Covering the whole range of edge amplitudes ensures that
    # ROC and PR curves are complete.
    zvec = [-1] + list(zvec) + [max(zvec) + 1]
    for z in range(1, len(zvec)):
        Atmp[abs(Atmp) <= zvec[z]] = 0
        aests.append(round(Atmp, ndigits=5))

    return aests
