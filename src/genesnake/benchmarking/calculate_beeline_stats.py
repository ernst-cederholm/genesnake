"""
Calculates network benchmarking metrics (Early Precision Rate and Top-K Accuracy).
Follows conventions from the dSPIN preprint (doi: 10.1101/2023.04.19.537364).
"""

from typing import Optional

import numpy as np
import genesnake as gs


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
    
    Args:
        inferred_network: The continuous edge weight matrix of the predicted GRN.
        reference_network: The boolean ground-truth edge matrix.
        fraction_of_k: The fraction of total true edges to evaluate.
        top_k: An explicit number of top edges to evaluate (overrides fraction_of_k).
        
    Returns:
        The precision (True Positives / Predicted Positives) among the top K edges.
    """
    reference_bool = np.asarray(reference_network, dtype=bool)
    
    # 1. Determine the integer K
    if top_k is not None:
        k = int(top_k)
    else:
        k = int(round(reference_bool.sum() * fraction_of_k))
        
    if k <= 0:
        return 0.0

    # 2. Flatten and get absolute values to evaluate connection strength
    abs_net_flat = np.abs(np.asarray(inferred_network)).flatten()
    n_elements = len(abs_net_flat)
    
    # Cap K if it exceeds the maximum possible edges
    k = min(k, n_elements)

    # 3. Find the cutoff threshold using O(N) partitioning instead of O(N log N) sorting
    cutoff_idx = n_elements - k
    cutoff = np.partition(abs_net_flat, cutoff_idx)[cutoff_idx]
    
    # 4. Create boolean mask of inferred edges that meet the cutoff
    inferred_bool = np.abs(inferred_network) >= cutoff
    
    # 5. Calculate statistics
    true_positives = (inferred_bool & reference_bool).sum()
    predicted_positives = inferred_bool.sum()

    if predicted_positives == 0:
        return 0.0
        
    return float(true_positives / predicted_positives)