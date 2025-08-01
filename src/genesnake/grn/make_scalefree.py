import numpy as np
import pandas as pd
import random

def make_scalefree(N, S, alpha=1.2, pasign=0.62):
    """
    Create a scale-free network with N nodes and specific sparsity.
    This version allows creating controllable in/out degree closer to real power-law distributions.

    Parameters:
    - N (int): number of nodes in the network
    - S (float): sparsity of the network
    - alpha (float): power value in the power law distribution (default 1.2)
    - pasign (float): probability of activation sign (default 0.62)

    Returns:
    - A (numpy.ndarray): adjacency matrix representing the network
    """
    spar = S - 1  # due to self-loops
    gns = [f"G{i+1}" for i in range(N)]  # list of gene names as G1, G2, ..., GN

    # Initial link
    lnk0 = random.sample(gns, 2)  # draw two genes to form the first seed link
    
    # Initialize network (edge list)
    net0 = [(lnk0[0], lnk0[1])]
    
    # Initial degree calculation
    dgr = 0
    i = 1

    while dgr < spar:
        # Draw the "rich" gene where the next will attach
        nds = [n for edge in net0 for n in edge]  # flatten list of nodes (from and to)
        unique_nodes, counts = np.unique(nds, return_counts=True)
        
        Pd = (counts / np.sum(counts)) ** alpha  # power-law distribution
        Pd /= np.sum(Pd)  # Normalize to sum to 1
        
        cp = np.concatenate(([0], np.cumsum(Pd)))
        rand_val = random.random()
        grich_index = np.searchsorted(cp, rand_val) - 1  # find "rich" gene based on probability
        grich = unique_nodes[grich_index]

        # Check genes that are not attached to any node
        gdif = list(set(gns) - set(unique_nodes))
        
        if len(gdif) > 0:
            probs2 = np.ones(len(gdif)) / len(gdif)  # Equal probability for unconnected nodes
        else:
            gdif = gns
            probs2 = np.ones(N) / N  # Equal probability for all nodes
            probs2[gns.index(grich)] = 0  # Exclude "rich" gene to avoid self-loops
        
        probs2 /= np.sum(probs2)  # Normalize probabilities
        cp2 = np.concatenate(([0], np.cumsum(probs2)))
        rand_val2 = random.random()
        gnext_index = np.searchsorted(cp2, rand_val2) - 1  # Find next gene to attach
        gnext = gdif[gnext_index]

        # Add edge only if it's a unique connection (to avoid duplicate edges)
        if (grich, gnext) not in net0 and (gnext, grich) not in net0:
            if random.random() < 1:  # Control edge direction (default: BA algorithm)
                net0.append((grich, gnext))
            else:
                net0.append((gnext, grich))

        # Update degree distribution
        nds0 = [n for edge in net0 for n in edge]
        unique_nodes0, counts0 = np.unique(nds0, return_counts=True)
        dgr = np.sum(counts0) / N / 2  # Average degree
        
    # Build adjacency matrix A
    A = np.zeros((N, N), dtype=int)
    node_to_index = {gns[i]: i for i in range(N)}  # Map gene names to matrix indices

    for (from_node, to_node) in net0:
        # Get indices from node names
        nr = node_to_index[from_node]
        nc = node_to_index[to_node]
        
        # Randomize the activation sign based on TRRUST probability
        if random.random() <= pasign:
            val = 1
        else:
            val = -1
        
        # Set the adjacency matrix value
        A[nc, nr] = val  # Note: outdegree nodes are in columns
    
    # Set diagonal to -1 (avoid self-loops)
    np.fill_diagonal(A, -1)

    gene_names = ["G"+str(i+1) for i in range(N)]
    network = pd.DataFrame(
        data = A.T,
        index = gene_names,
        columns = gene_names
        )

    return network


# Example parameters
#N = 10  # number of nodes
#S = 3  # sparsity of the network

# Generate scale-free network
#A = make_scalefree(N, S)

# Print adjacency matrix
#print(A)