"""
script for generating a random k-out graph with preferential attachment 
"""

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
import random

def make_DAG(nodes,degree,self_loops=False,single_comp=True,sign_bias=0.6):
    """
    Function for generating random DAG, this is not a super realistic GRN model but will work well for small systems <20 or so genes 
    after that the model will still work but may not be biologically realistic. 
    INPUT: 
        nodes - int for the number of nodes in the system.
        degree - int for the out-degree each node should have
        self_loops - optional - bool for if self loops are allowed or not. 
        single_comp - option to add extra edges to make sure any isolated node is connected to the system 
                      this is done by simply adding a single edge between the isolated node n from n+1
    OUTPUT: 
        network - a graph in adjacency matrix format stored in a pandas dataframe with each gene named G1..Gn. 
    """
    
    # start by building a lower triangle matrix,
    # this will always be a DAG as no edge can go
    # up to a number higher than it's own node
    net = np.zeros((nodes,nodes))
    while np.count_nonzero(net)/nodes<degree:
        n1 = random.randint(0,nodes-1)
        n2 = random.randint(0,nodes-1)
        
        # pick a sign
        r = random.random()
        if r<sign_bias:
            sign = 1
        else:
            sign = -1
            
        # and then a direction for the edge 
        if n1>n2:
            net[n1,n2] = sign
        elif n1<n2:
            net[n2,n1] = sign
    
    # if self loops add a -1 to all loops
    if self_loops:
        np.fill_diagonal(net,-1)
        
    if single_comp:
        # if we want to make sure the graph is fully connected
        # firt creata a networkx graph of it
        G = nx.from_numpy_array(net)
        # get all components the graph consists of 
        comps = list(nx.connected_components(G))
        # if there is more than 1 component
        if len(comps) > 1:

            # first sort over the comps so the largest is first
            comps = sorted(comps, key=lambda x:len(x),reverse=True)
            
            # loop over all and merge them
            for i in range(1,len(comps)):
                # select the largest node in the isolated component
                max_comp = np.array(list(comps[0]))
                max_n = max(list(comps[i]))
                # get the genes that can regulate the isolated component
                cands = np.where(max_comp>max_n)[0]
                
                # make a random sign
                r = random.random()
                if r<sign_bias:
                    sign = 1
                else:
                    sign = -1
                # if by chanse no candidates exist instead reverse the order of the edg
                if len(cands)>0:
                    # select a random node from the list of candidates
                    new_reg = random.choice(max_comp[cands])
                    
                    # and add a new edge with a random sign
                    net[new_reg,max_n] = sign
                else:
                    new_reg = random.choice(max_comp)
                    net[max_n,new_reg] = sign
    
    network = pd.DataFrame(net)
    
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
