import sys
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
    aests = []
    if (zvec_auto):
        zvec = np.linspace(0, max(abs(A)), 100)
    else:
        ep = sys.float_info.epsilon
        zmin = np.abs(A[A > 0]).min().min() - ep
        zmax = np.abs(A[A > 0]).max().max() + ep * 10
        zvec = zvec * (zmax - zmin) + zmin

    Atmp = deepcopy(A)
    zvec = [-1] + list(zvec)
    for z in range(1, len(zvec)):
        # Add a cutoff that is smaller than 0, so that at least
        # one full network is returned //AB
        Atmp[abs(Atmp) <= zvec[z]] = 0
        aests.append(round(Atmp, ndigits=5))
    return aests
