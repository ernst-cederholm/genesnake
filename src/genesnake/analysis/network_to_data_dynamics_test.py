
import pandas as pd
import numpy as np

def _get_regulations(GRNmodel):
    """
    Function to create an demi-edge list from a network and the collaborative parameter in a geneSNAKE GRNmodel.
    """
    
    #first get relative change in the system by fold change.
    # not this is not log2 fold change as it does not matter
    fold_change_data = np.zeros((genes,genes*3))
    for i in range(GRNmodel.genes*3):
        for j in range(GRNmodel.genes):
            fold_change_data[j,i] = GRNmodel.noise_free_data.values[j,i]/GRNmodel.steady_state_RNA[j]
    fold_change_data = pd.DataFrame(fold_change_data,index=M.data.index,columns=M.data.columns)
        
    
    # as all edges and collaborative interactions are already stored in the alpha parameter it is used to find all regulations   
    for i in GRNmodel.parameters.alphas.keys():
        # the alpha is structured regulator_target so we seperate them that way
        reg_targ = i.split("_")
        regulators = reg_targ[0:-1]
        target = reg_targ[-1]
        alpha = GRNmodel.parameters.alphas[i]
        
        
        

def network_data_dyn_test(GRNmodel):
    """
    Function that attempts to estimate if the realtive change from RNA inital state in the GNR model
    and the final time point corresponds to the expected effect based on the network, perturbation
    and determined collaborative behaviour between regulators.
    It does this by looping over each gene in each experiment so for larger data it may be slow.
    NOTE the estimation is primarily just based on the direction of the change and not the value.
    INPUT:
        GNRmodel - a genesnake.GRNmodel with initial steady_state_RNA and experimental data.
    OUTPUT:
        edge_dyn - a per regulation (edges and collaborative) mask of True or False based on if the
                   effect seem to correspond to the expected sign. 
    """
    
    regulations = _get_regulations(GRNmodel)

