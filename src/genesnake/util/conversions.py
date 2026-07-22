
import numpy as np
import pandas as pd
from copy import deepcopy


def edgelist_to_matrix(
        regulators,
        targets,
        values = None,
        all_genes = None):

    from scipy.sparse import coo_array

    assert(len(regulators) == len(targets))
    if values is None:
        values = [1] * len(regulators)

    if all_genes is not None:
        unique_genes = sorted(np.unique(all_genes))
    else:
        unique_genes = sorted(
            np.unique(list(regulators) + list(targets))
            )
    n = len(unique_genes)
    index_dictionary = dict(zip(unique_genes, range(n)))
    iis = [index_dictionary[element] for element in regulators]
    jjs = [index_dictionary[element] for element in targets]

    sparse_array = coo_array((values, (iis, jjs)), shape = (n, n))
    numpy_array = sparse_array.toarray()
    df = pd.DataFrame(numpy_array)
    df.index = unique_genes
    df.columns = unique_genes

    return(df)




def matrix_to_edgelist(df):

    edgelist = []
    row_names = np.array(df.index).astype(str)
    col_names = np.array(df.columns).astype(str)
    data = np.array(df, order = 'C').astype(float)

    regulators = []
    targets = []
    values = []
    for ii in range(data.shape[0]):
        jjs = data[ii, :].nonzero()[0]
        regulators.append([row_names[ii]] * len(jjs))
        targets.append(col_names[jjs])
        values.append(data[ii, jjs])

    df = pd.DataFrame(
        dict(zip(
            ['regulator', 'target', 'value'],
            [
                np.concatenate(regulators),
                np.concatenate(targets),
                np.concatenate(values)
                ]
            )) 
        )
    return(df)




def harmonise_networks(network_matrices):

    tmp = []
    [tmp.extend(df.columns) for df in network_matrices]
    all_genes = sorted(np.unique(tmp))

    edge_lists = [matrix_to_edgelist(net) for net in network_matrices]

    harmonised_networks = [
        edgelist_to_matrix(
            regulators = l['regulator'],
            targets = l['target'],
            values = l['value'],
            all_genes = all_genes
            )
        for l in edge_lists
        ]
    return(harmonised_networks)



