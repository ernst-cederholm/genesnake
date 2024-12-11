
import numpy as np
from copy import deepcopy
def cutoff_handler(A, zvec = np.logspace(-5, 0, 100), zvec_auto=True):
    """
    Create N networks under various cutoffs using vector of so-called zeta values as thresholds.

    With the default zvec_auto = True, it picks representative cutoffs
    from the edge amplitude distribution.
    After picking other zetas, includes zetas such that at least
    one unmodified network and one full network are included in
    the output list of networks.

    INPUT:
        A
            A network in matrix form
        zvec
            zeta vector.
            Note that this one is relative. Final zeta values are
            scaled with maximum and minimum edge values
        zvec_auto
            Pick zetas from the sorted list of all network edge amplitudes
            at index wise regularly spaced intervals. The sorted list
            can be seen as the empirical cumulative distribution function
            of edge amplitudes. With cutoffs adapted to the distribution
            of the edge amplitudes, there is a good chance they
            will lead to an informative range of networks.
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
