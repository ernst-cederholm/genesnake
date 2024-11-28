import pandas as pd
import numpy as np 

def numpy_to_adjacency(net,isolates=False):
    """
    function to transform a numpy edgelist in to an adjacency matrix
    INPUT: 
        net - an edgelist in a numpy.array with the order regulator target weight (weight can be 1s)
        isolates - [optional] - The number of genes in the system not listed in the edgelist to be added in the matrix. 
                                Will be added to the end of the matrix as Gn+1..GN+1+isolates.
    OUTPUT: 
        adj_mat - a pandas adjacency matrix that can be feed to geneSNAKE functions
    """
    
    # test if net in a numpy.arrya and print warning otherwise
    if not isinstance(net, np.ndarray):
        raise TypeError("Input to numpy_to_adjacency must be numpy array")
    
    # get all names of genes this is done to ensure no edge ends up wrong
    gene_names = np.unique(net[:,0:2]).astype(int).tolist() # take 0:2 as it is non-inclusive giving us the first 2 columns
    
    # we then fill any isolated gene (not in edge list) in with a pseduo name
    if isolates:
        end = length(gene_names)+1
        while len(gene_names) < len(gene_names)+isolates:
            gene_names = gene_names.append("G"+str(end))
            end+=1
        
    # from the gene names create an empty pandas dataframe 
    adj_mat = pd.DataFrame(index=gene_names,columns=gene_names)
    
    # and fill it in based on gene1 gene2 name
    for i in range(net.shape[0]):
        # note that pandas works with a column row notation
        # so that net[i,1],net[i,0] here sorts the network from
        # row to column. meaning a regulator is on row and target in column
        # get the row and column IDs and convert to int if float
        row = net[i,0]
        col = net[i,1]
        
        if isinstance(row,float):
            row = int(row)
        if isinstance(col,float):
            col = int(col)
        
        # the if statement here is used to select the absolute maximum if more than one copy of an edge exists
        if pd.isna(adj_mat[col][row]):
            adj_mat[col][row] = net[i,2]
        elif abs(net[i,2]) > abs(adj_mat[col][row]):
            adj_mat[col][row] = net[i,2]
        
    # we then remove NaN values as they will cause issues later on 
    adj_mat = adj_mat.fillna(0)
        
    return adj_mat 


def pandas_to_adjacency(net,isolates=None):
    """
    function to transform a pandas edgelist in to an adjacency matrix
    INPUT: 
        net - an edgelist in a pandas.DataFrame with the order regulator target weight (weight can be 1s)
        isolates - [optional] - The number of genes in the system not listed in the edgelist to be added in the matrix. 
                                Will be added to the end of the matrix.
    OUTPUT: 
        adj_mat - a pandas adjacency matrix that can be feed to geneSNAKE functions
    """
    
    # test if net in a pd.dataframe and print warning otherwise
    if not isinstance(net, pd.DataFrame):
        raise TypeError("Input to pandas_to_adjacency must be pandas dataframe")

    # define a custom functon that finds the absolute maximum of a given dataframe to merge on
    # this is to remove any redundant edges in the system
    absmax=lambda x: max(x.min(), x.max(), key=abs)
    # then use the built in function crosstab to take the edge list and make it in to a square matrix
    # use crosstab as this lets us decide how to handle if the same edge exists more than once.
    adj_mat = pd.crosstab(index=net.iloc[:,0], columns=net.iloc[:,1],values=net.iloc[:,2], aggfunc=absmax)
    
    # as crosstab only adds rows and columns for each element existing in the original column add in empty rows/columns
    # and place them in the order they are found in the column or index names respecitvely
    row_ind = adj_mat.index.union(adj_mat.columns)
    col_ind = adj_mat.columns.union(adj_mat.index)
    
    adj_mat = adj_mat.reindex(index=row_ind, columns=col_ind, fill_value=0)
    
    # we then remove NaN values as they will cause issues later on 
    adj_mat = adj_mat.fillna(0)
    
    # drop the names of the columns we now use for index and column names as it is no longer relevant 
    adj_mat.index.name = None
    adj_mat.columns.name = None
    
    return adj_mat
    
