"""
functions realted to creating network with feed-forward loops in them
"""

from . import auxilary_functions as functions
from .generation_algorithm import *
import pandas as pd
import numpy as np
import json


def make_FFLATnetwork(**kvargs):
    """
    function for generating a network using the FFLAT model
    """
    #!# improve documentation when it is clear what each element does. 

    #!# this is dumb rewrite so that it takes imput in a better way and then strores that in a dict instead of what ever this is. Thomas to Thomas
    
    # start up a dict with all the keys and default values this is done both to have default
    # values and to make sure nothing strange happens if one value is not in the config file
    settings = {"RANDOM_SEED":np.random.randint(1,100), "FFL_ENRICHED":1, "SPARSITY":2.328, "NETWORK_SIZE":100, "N_CORES_TO_USE":-1, "SHUFFLED":0, "OUTPUT":"adj_list", "NO_CYCLES":1}

    # if there is a file to load configs from do that  
    if "LOAD" in kvargs.keys() or "load" in kvargs.keys():
        file = open(kvargs["load"],"r")
        configs = json.load(file)
        file.close
        # loop over all keys in the json file and add them to the settings dict. 
        for key in configs.keys():
            if key.upper() in settings.keys(): # use .upper to allow users to have small or large letters in save
                settings[key] = configs[key]
            else:
                print ("WARNING: Found unknown setting in config file: "+str(kvargs["load"])+".\nFor a list of avalible options see documentation of genesnake.network.make_FFLATnetwork, this can be done e.g. with print(gs.network.make_FFLATnetwork.__doc__) or any other place the documentation is avalible.\n")
        
    # then add all setting from the kvargs input that isn't load
    for key in kvargs.keys():
        if key.upper() == "LOAD":
            continue
        elif key.upper() in settings.keys(): # use .upper to allow users to have small or large letters in save
            settings[key.upper()] = kvargs[key]
        else:
            raise TypeError("Unknown input option in genesnake.network.make_FFLATnetwork. For avalible input options see documentation at e.g print(gs.network.make_FFLATnetwork.__doc__) or any other place the documentation is avalible")
        
    # this part needs to be fixed to either take kvargs input or not have all these
    # extra options 
    interaction_matrix = functions.get_interaction_matrix(settings)
    
    motifs = functions.motif_search(settings, interaction_matrix, batch_size=10000, load=True, dump=False)
    motifs_orig = motifs["030T"]
    
    ffl_nodes = list(set(sum([list(map(int, x.split("_"))) for x in motifs_orig], [])))
    interaction_matrix_ffl = np.zeros((len(ffl_nodes), len(ffl_nodes)))
    
    for motif in motifs_orig:
        motif = functions.split_motif(motif)
        motif_new = list(ffl_nodes.index(x) for x in motif)
        interaction_matrix_ffl[np.ix_(motif_new, motif_new)] = interaction_matrix[np.ix_(motif, motif)]
    interaction_matrix_ffl.shape, interaction_matrix_ffl.sum()
        
    #Load motif space
    motifs_network = functions.build_vmn(motifs_orig, verbose=True, load=True, dump=False)
    
    # what are all the settings here?
    artificial_matrix_ffl = generate_artificial_network(
                        interaction_matrix, settings, random_seed = np.random.randint(1,100),
                        motifs=motifs_orig, motifs_network=motifs_network,
                        reference_matrix=interaction_matrix_ffl, network_size=int(settings["NETWORK_SIZE"]))
    
    # creata a numpy array to store the data in 
    tmpnet = np.zeros((int(settings["NETWORK_SIZE"]),int(settings["NETWORK_SIZE"])))
    for i in artificial_matrix_ffl:
        # randomly sign the network
        #!# this needs to be improved still! 
        p = np.random.random()
        if p > 0.5:
            tmpnet[i[0],i[1]] = 1
        else:
            tmpnet[i[0],i[1]] = -1
                    
    # then make up gene names by adding G1..Gn
    names = ["G"+str(i+1) for i in range(int(settings["NETWORK_SIZE"]))]

    # add weights from a uniform distribution
    tmpnet = np.multiply(tmpnet,np.random.uniform(0.2,1,tmpnet.shape))
    
    # and make a pandas datafram to return this information to
    network = pd.DataFrame(tmpnet,index=names,columns=names)
    
    return network
