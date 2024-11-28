"""
script for generating a random k-out graph with preferential attachment 
"""

import networkx as nx
import pandas as pd
import numpy as np

def make_random(nodes,degree,alpha=0.9,self_loops=False):
    """
    Function for generating random k-out graphs with preferential attachment acts as a wraper of the networkx function 
    random_k_out_graph. It is included as a way to easily get a networkx network in to a genesnake format but any part of this 
    function can easily be implemented seperatly for further options. The code here can also be used as an example in how to 
    easiest get a networkx model in to genesnake format (pandas adjacency matrix). 
    INPUT: 
        nodes - int for the number of nodes in the system.
        degree - int for the out-degree each node should have
        alpha - optional - float used for selecting the node a new edge should be added to. A higher number gives a more uniform distrubution. Default 0.9 and a lower number means a node is more likely to be selected as the indegree increases. 
        self_loops - optional - bool for if self loops are allowed or not. 
    OUTPUT: 
        network - a graph in adjacency matrix format stored in a pandas dataframe with each gene named G1..Gn. 
    """
    
    # round the degree as it has to be an int
    degree = int(round(degree))
    
    # make a random graph
    # Note that degree is x2 and half of them are then droped to make it directed 
    G = nx.random_regular_graph(degree*2,nodes)

    # and move it to pandas
    network = nx.to_pandas_adjacency(G)

    # generate names for each gene 
    names = ["G"+str(i+1) for i in range(nodes)]
    
    # add these to the network dataframe
    network.index = names
    network.columns = names
    
    # remove all symetric links
    for i in range(network.shape[0]):
        ind = np.where(network.values[i,:])[0]
        for j in ind:
            if network.values[i,j] != 0 and network.values[j,i] != 0:
                r = np.random.random()
                if r > 0.5:
                    network.values[i,j] = 0
                else:
                    network.values[j,i] = 0
    
    # scale the network to be 1.0 for all edges to fit the genesnake framework
    network = network.where(network<1.0,1.0)
    # and then add weights so we have more control over the range
    network = np.multiply(network,np.random.uniform(0.2,1,network.shape))
    # and add random sign in range +/-1
    sign  = np.random.randint(2,size=(nodes,nodes))*2-1
    # add sign to the network 
    network = network.multiply(sign,axis=0)
    
    # if self loops add a -1 to all loops
    if self_loops:
        diag = [np.random.uniform(np.max(np.abs(network.values))/2,np.max(np.abs(network.values)))*-1 for i in range(len(names))]
        np.fill_diagonal(network.values,diag)
        
    # and remove -0.0s for estetics (this step does not affect anything and is added due to personal preferense)
    network = network.mask(network==0.0,0.0) 
    
    return network.T
