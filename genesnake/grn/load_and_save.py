# file containing functions for exporting and importing GRNs. Primarily uses pandas for all loading and saving
import networkx as nx 
import pandas as pd
import numpy as np
import random


def to_edgelist(network,file,sep=","):
    """
    Function for exporting a network as an edge list. Not realy dependent on any geneSNAKE format
    instead takes a adjacencymatrix from pandas and transforms it to an edge list and saves to file.
    INPUT:
        - network - the network to save, must be in pandas.dataframe format and contain index and
                    column names representing each gene
        - file - the file name to save the network to
        - sep - the character used to seperate the columns in the output 
    """
    
    # start by transforming the network in to an adjacencymatrix
    edglst = network.rename_axis('Source').reset_index().melt('Source', value_name='Weight', var_name='Target').reset_index(drop=True)
    # remove any column without an edge
    edglst = edglst[edglst["Weight"]!=0].reset_index(drop=True)
    
    # round the weight column as we don't need 16 bits in the file
    edglst["Weight"] = edglst["Weight"].round(2)
    
    # finally save the edge list
    edglst.to_csv(file,sep=sep,index=False)


def load(file,kind="edgelist",sep=None,rownames=None,header=0,shuffle=False):
    """
    Function for loading a saved network in to the expected format for geneSNAKE.
    INPUT:
        - file - the path and file of the file to be loaded.
        - kind - the type of file to expect, either adjacency matrix or edgelist. Accepted options are [edgelist, adjacency]
        - sep - the character used to seperate each column
        - rownames - which column contains the row names for adjacency matrix. if None the header will be used
        - header - which row contains the header, should be gene symbols for adjacency matrix.
                   If None row names will be used instead.
                   For edgelist this indicates that the first row should not be used when generating the network. 
                   NOTE: if both rownames and header is None names will be generated using G1..Gn.
        - shuffle - randomly sort the genes and rename them G1..Gn in the network while maintaining the
                     connections. This is useful if you for some reason want to anonymize the network.
                     E.g. for public benchmarks that uses a known GRN.
    OUTPUT:
        - grn - a pd.dataframe of the adjacency matrix of the network suitable for the geneSNAKE package. 
    """
    
    if kind == "edgelist":
        # read in the file and turn it to an adjacency matrix 
        grn = pd.read_csv(file,sep=sep,index_col=rownames,header=header)
        grn = grn.pivot(index=grn.columns[0],columns=grn.columns[1],values=grn.columns[2])
        grn = grn.fillna(0)
        
        # remove the name of column and header
        grn.index.name = None
        grn.columns.name = None
        
        # change the sign symbols to numbers if they are there 
        grn = grn.replace("+",1)
        grn = grn.replace("-",-1)
        
        # finally add any column or index that is not there due to only being a regulator or target
        missing_cols = [i for i in grn.index if i not in grn.columns]
        for col in missing_cols:
            grn[col] = 0
        missing_inds = [i for i in grn.columns if i not in grn.index]
        for ind in missing_inds:
            grn.loc[ind] = 0
            
        # then sort the dataframe on the index just so that they are the same
        grn = grn[grn.index]
        
    elif kind == "adjacency":
        grn = pd.read_csv(file,index_col=rownames,header=header,sep=sep)
        if rownames==None and header!=None:
            grn.index = grn.columns
        elif rownames!=None and header==None:
            grn.columns = grn.index
        else:
            names = ["G"+str(i+1) for i in range(len(grn))]
            grn.index = names
            grn.columns = names
    else:
        raise ValueError("genesnake.grn.load got an unknown GRN format, allowed formats are adjacency and edgelist.")
    
    if shuffle==True:
        # draw a random order 
        new_order = grn.columns.tolist()
        random.shuffle(new_order)
        # then change the order of everything to follow this new order 
        grn = grn.reindex(new_order)
        grn = grn[new_order]
        
        # create new names that goes from 1..n
        names = ["G"+str(i+1) for i in range(len(grn))]
        grn.index = names
        grn.columns = names
        
    return grn


def from_networkx(network, undirected=False,sign_bias=0.6,add_name=True,weight="add"):
    """
    Function for moving a networkx graph in to a pandas dataframe for use in geneSNAKE.
    Primarily just wraps the networkx.to_pandas_adjacency function. It will add sign to the network
    along with forcing directionallity if the undirected option is used. 
    INPUT:
        - network - the networkx graph to convert
        - undirected - boolean if the original network were undirected randomly remove the edge in on direction
                       to force a directed nature in to the system.
        - sign_bias - a float that determine how likely it is that a sign is an activator, and thus not
                      inhibitory. If you wish to just not change sign set sign_bias=1 as this will maintain
                      the original weight. 
        - add_name - boolean that controls if names should be generated for the nodes or not.
                     will create names in format G1..Gn
        - weight - a string where the weights stored in the networkx graph or \"add\" for adding new weights
                   to the network. 
    OUTPUT:
        - grn - the GRN ready to be used in the GRNmodel.  
    """
    # first move the graph to a pandas dataframe 
    if weight=="add":
        grn  = nx.to_pandas_adjacency(network)
        # add weights from a uniform distribution
        grn = np.multiply(grn,np.random.uniform(0.2,1,grn.shape))
    else:
        grn  = nx.to_pandas_adjacency(network,weight=weight)
    # if adding names generate and add these 
    if add_name==True:
        names = ["G"+str(i+1) for i in range(len(grn))]
        grn.index = names
        grn.columns = names
        
    # for each row find the nonzero element and add sign
    for i in range(grn.shape[0]):
        ind = np.where(grn.loc[grn.index[i]])[0]
        for j in ind:
            r = random.random()
            if r>sign_bias:
                grn.iloc[i,j] = grn.iloc[i,j]*-1
            # if we are adding direction by removing edges do that here
            if undirected==True and grn.iloc[i,j]!=0 and grn.iloc[j,i]!=0:
                r = random.random()
                if r>0.5:
                    grn.iloc[i,j] = 0
                else:
                    grn.iloc[j,i] = 0
              
    return grn

def from_numpy(array):
    """
    Simple function that takes a 2D numpy array, transform it in to a pandas.dataframe
    and add names in format G1..Gn to make it easy to use numpy in genesnake.GRNmodel creation.
    INPUT:
        - array - the numpy array to use.
    OUTPUT:
        - grn - the network in a pandas adjacency matrix with names
    """
    names = ["G"+str(i+1) for i in range(len(array))]
    grn = pd.DataFrame(array,index=names,columns=names)
    
    return(grn)


def to_networkx(network):
    """
    wraper function that uses networkx.from_pandas_adjacency to move a GRNmodel suiatble network to networkx.
    NOTE this function only exists to help people less familiar with pandas and networkx to easily
    move between the two.  
    INPUT:
        - network - the network to convert to networkx
    OUTPUT:
        - graph - the networkx graph object made. 
    """
    graph = nx.from_pandas_adjacency(network,create_using=nx.DiGraph())
    
    return graph
    
