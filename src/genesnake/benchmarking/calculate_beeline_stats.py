"""
Calculates network benchmarking metrics (Early Precision Rate and Top-K Accuracy).
Follows conventions from the dSPIN preprint (doi: 10.1101/2023.04.19.537364).
"""

from typing import Optional

import numpy as np
import genesnake as gs

# Weights at or below this magnitude are treated as "no prediction".
ZERO_WEIGHT_TOL: float = 1e-12


def early_precision_rate(
    inferred_network: np.ndarray, 
    reference_network: np.ndarray,
    fraction_of_k: float = 1.0,
    top_k: Optional[int] = None
) -> float:
    """
    Calculates the Early Precision Rate (EPR) by normalizing the Top-K accuracy
    against the Expected Random Model AUPR (ERMA).
    
    Args:
        inferred_network: The continuous edge weight matrix of the predicted GRN.
        reference_network: The boolean ground-truth edge matrix.
        fraction_of_k: The fraction of total true edges to evaluate (default 1.0).
        top_k: An explicit number of top edges to evaluate (overrides fraction_of_k).
        
    Returns:
        The EPR value as a float.
    """
    top_k_acc = top_k_accuracy(
        inferred_network=inferred_network, 
        reference_network=reference_network,
        fraction_of_k=fraction_of_k,
        top_k=top_k
    )

    # calculate_ERMA divides by the network size, so guard the empty case.
    if np.asarray(reference_network).size == 0:
        return 0.0

    erma = gs.benchmarking.calculate_ERMA(reference_network)
    
    if erma <= 0.0:
        return 0.0
        
    return float(top_k_acc / erma)


def top_k_accuracy(
    inferred_network: np.ndarray, 
    reference_network: np.ndarray,
    fraction_of_k: float = 1.0,
    top_k: Optional[int] = None
) -> float:
    """
    Calculates the precision of the network using only the strongest K predictions.

    Zero weights (|w| <= ZERO_WEIGHT_TOL) are not predictions and are never counted,
    even when the network has fewer than K non-zero weights. In that case the
    precision is computed over the non-zero predictions only (TP / PP), so sparse
    methods are neither rewarded nor penalised for leaving edges at zero.

    Args:
        inferred_network: The continuous edge weight matrix of the predicted GRN.
        reference_network: The boolean ground-truth edge matrix.
        fraction_of_k: The fraction of total true edges to evaluate.
        top_k: An explicit number of top edges to evaluate (overrides fraction_of_k).
        
    Returns:
        The precision (True Positives / Predicted Positives) among the top K
        non-zero edges, or 0.0 if there are no such edges.
    """
    reference_bool = np.asarray(reference_network, dtype=bool)

    # 1. Determine the integer K
    if top_k is not None:
        k = int(top_k)
    else:
        k = int(round(reference_bool.sum() * fraction_of_k))

    # 2. Flatten and get absolute values to evaluate connection strength
    abs_net = np.abs(np.asarray(inferred_network))
    abs_net_flat = abs_net.flatten()
    n_elements = len(abs_net_flat)

    # Cap K if it exceeds the maximum possible edges
    k = min(k, n_elements)
    if k <= 0:
        return 0.0

    # 3. Find the cutoff threshold using O(N) partitioning instead of O(N log N) sorting
    cutoff_idx = n_elements - k
    cutoff = np.partition(abs_net_flat, cutoff_idx)[cutoff_idx]

    # 4. Predicted edges: at or above the cutoff AND non-zero. With fewer than K
    #    non-zero weights the cutoff is 0, and the second condition keeps true
    #    zeros out of predicted_positives.
    inferred_bool = (abs_net >= cutoff) & (abs_net > ZERO_WEIGHT_TOL)

    # 5. Calculate statistics
    true_positives = (inferred_bool & reference_bool).sum()
    predicted_positives = inferred_bool.sum()

    if predicted_positives == 0:
        return 0.0
        
    return float(true_positives / predicted_positives)


def precision_at_k(
        inferred_network,
        reference_network,
        fraction_of_P = 1):
    """
    For background, see the docstring of top_k_accuracy.

    Another choice for handling the problem when PP < K is to choose to not
    score PN in the top K list, and thus setting TN = 0. Then,
    metric = (TP + 0) / K = TP / K.
    The bad part is that this metric is neither accuracy nor precision.
    A good part is that it handles the problem of TN in such
    a way that one avoids the issues with using accuracy for class imbalanced
    problems. One can not trivially game this metric for sparse problems, and
    so it is infomative in that sense.
    This metric seems to be commonly known as 'precision at K',
    so I adopted that name to be as consistent and conventional as possible.

    For PP > K, this metric equals the precision of the top K predictions.
    This is probably where it gets it's name from, and I am guessing that
    people who use it also rarely encounter PP < K. For PP > K, it's also
    equal to top_k_accuracy.

    For PP < K, this metric will be < precision. The difference will increase
    with K - PP. One way to think about it is that the metric penalises
    models that are too sparse, by substituting a score 0 for the K - PP
    predictions they didn't make.
    Thus, sparse models that achieves high precision in exactly
    the way they are designed to, by being sparse, will not score well
    in this metric, despite the metric having precision in the name, a metric
    that is commonly used and suitable for sparse problems and predictions.
    This risks masking interesting performances by sparse
    models that one would like to see and know about.

    Being a "top K" metric, the intuition is that it's about the "early"
    performance of models, a small subset of the most confident predictions.
    Depending on how K is set, that can probably be the case. As we've
    seen it used in GRN inference though, K is usually set to P, the number of
    positive / existing edges in the reference network.
    That choice of K means that a model needs both recall 1 and precision 1 to
    reach 1 in precision at K.
    In fact, precision = recall, because for top K metrics, and with K = P,
    precision = TP / PP = TP / K = TP / P = recall
    Fixing precision = recall, and thus requiring good recall for good score
    does not match the intuition about the metric having to do with "early"
    performance. One perspective might be that the notion of "early" performance
    is inherently tied to the precision of the most confident predictions,
    and this metric is not precision. Of course, the blame for this might
    be on the selection of K rather than the metric itself. For K << P,
    high precision at K does not require high recall.

    Since precision at K with K = P is as much about recall
    as precision (exactly actually), it seems similar to other well known metrics,
    like AUPR and f1 scores. The practical tests I did also showed that
    precision at K behaves similar to AUPR. This makes is less useful in
    practice, conveying similar information.

    So, the paragraphs above explain why I left the original
    top K accuracy untouched (despite its admittedly poor name), and
    instead introduced this metric as an option alongside it.
    """
    reference_network = reference_network.astype(dtype=bool)
    K = sum(reference_network)
    K = round(K * fraction_of_P)

    if K == 0:
        import warnings
        warnings.warn('There are no edges in the reference network')
        p_at_k = np.nan
        return p_at_k

    tmp = inferred_network
    iis = sorted(range(len(tmp)), key = lambda i: tmp[i], reverse = True)
    ref = reference_network[iis][:K]
    inf = inferred_network[iis][:K].astype(dtype = bool)

    TP = sum(inf & ref)
    p_at_k = TP / K

    return p_at_k
