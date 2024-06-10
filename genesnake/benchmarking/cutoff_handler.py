import sys
import numpy as np
def cutoff_handler(A, zvec = np.logspace(-5, 0, 100), zvec_auto=False):
    """
    Create N networks under various cutoffs using vector of so-called zeta values as thresholds
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
        zmin = min(abs(A[A > 0])) - ep
        zmax = max(abs(A[A > 0])) + ep * 10
        zvec = zvec * (zmax - zmin) + zmin
    for z in range(1, len(zvec)):
        Atmp = A
        Atmp[abs(Atmp) <= zvec[z]] = 0
        aests.append(round(Atmp, ndigits=5))
    return aests