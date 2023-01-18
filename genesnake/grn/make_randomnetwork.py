"""
script for generating a random k-out graph with preferential attachment 
"""

import networkx as nx
import pandas as pd
import numpy as np

def make_random(nodes,degree,alpha=0.9,self_loops=True):
    """
    Function for generating random k-out graphs with preferential attachment acts as a wraper of the networkx function 
    random_k_out_graph. It is included as a way to easily get a networkx network in to a genesnake format but any part of this 
    function can easily be implemented seperatly for further options. The code here can also be used as an example in how to 
    easiest get a networkx model in to genesnake format (pandas adjacency matrix). 
    INPUT: 
        nodes - int for the number of nodes in the system.
        degree - int for the outdegree each node should have 
        alpha - optional - float used for selecting the node a new edge should be added to. A higher number gives a more uniform distrubution. Default 0.9 and a lower number means a node is more likely to be selected as the indegree increases. 
        self_loops - optional - bool for if self loops are allowed or not. 
    OUTPUT: 
        network - a graph in adjacency matrix format stored in a pandas dataframe with each gene named G1..Gn. 
    """

    # make a random graph 
    G = nx.random_k_out_graph(nodes,degree,alpha,self_loops=self_loops)
    # and move it to pandas 
    network = nx.to_pandas_adjacency(G)
    
    # generate names for each gene 
    names = ["G"+str(i+1) for i in range(nodes)]
    
    # add these to the network dataframe
    network.index = names
    network.columns = names
    
    # scale the network to be 1.0 for all edges to fit the genesnake framework
    network = network.where(network<1.0,1.0) 
    # and add random sign in range +/-1
    sign  = np.random.randint(2,size=(nodes,nodes))*2-1
    # add sign to the network 
    network = network.multiply(sign,axis=0)
    
    # and remove -0.0s for estetics (this step does not affect anything and is added due to personal preferense)
    network = network.mask(network==0.0,0.0) 
    
    return network
