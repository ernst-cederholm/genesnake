"""
A collection of functions that are used to randomly generate parameters for the modle.py script.
Note that they are not imported in to the data namespace only the model class. 
"""

import pandas as pd
import numpy as np
import itertools

class pargen:
    
    
    def make_start_rna(genes=None,high_exp=1,low_exp=0.1):
        if genes is None:
            raise ValueError("Attempted to generate parameters for None object in genesnake.modle.parameters")
        
        A= np.round(np.random.uniform(low=low_exp,high=high_exp,size=(len(genes))),2).tolist()
        return A
    
    def make_start_prot(genes=None,high_exp=1,low_exp=0.1):
        if genes is None:
            raise ValueError("Attempted to generate parameters for None object in genesnake.modle.parameters")
        
        return np.round(np.random.uniform(low=low_exp,high=high_exp,size=(len(genes))),2).tolist()

    
    def make_time(genes=None,time_points=10000,end_time=10):
        if genes is None:
            raise ValueError("Attempted to generate parameters for None object in genesnake.modle.parameters")
        # make the timepoints in a linspace between 0 and max
        # add +1 to number of time points as the 0th time point will be no change
        return np.linspace(0,end_time,time_points+1).tolist()
    
    def make_max_rna_exp(genes=None,Emax=1,Emin=1):
        if genes is None:
            raise ValueError("Attempted to generate parameters for None object in genesnake.modle.parameters")
        
        return np.round(np.random.uniform(low=Emin,high=Emax,size=(len(genes))),2).tolist()
    
    
    def make_max_prot_exp(genes=None,Emax=1,Emin=1):
        if genes is None:
            raise ValueError("Attempted to generate parameters for None object in genesnake.modle.parameters")
        
        return np.round(np.random.uniform(low=Emin,high=Emax,size=(len(genes))),2).tolist()
    
    
    def make_hill_coef(genes=None,Emax=4,Emin=2):
        if genes is None:
            raise ValueError("Attempted to generate parameters for None object in genesnake.modle.parameters")
        
        return np.round(np.random.randint(low=Emin,high=Emax+1,size=(len(genes))),2).tolist()
    
    
    def make_combinations(network=None,max_comb=2,comb_prob=0.3,names=None):
        """
        Function for drawing all possible combinations of genes regualting the same 
        targer in max_comb sized groups and storing this in a 
        dictonary. This is used for model generation in geneSNAKE.model.make_model. 
        Note that only genes of the same regulatory sign can act as a group. 
        Meaning inhibitory genes and activating genes can not enhance each other. 
        INPUT:
            network - the network to draw AND/OR combinatory gene regulation from in either numpy or pandas format
            max_comb -[optional]- the maximum number of genes that can act as a regualtory grouping
            comb_prob -[optional]- the probability of genes acting as and AND (all genes in the 
                        group needed) over an OR group (all genes acting independent)
            names -[optional]- a list of gene names to be used for the combination. If network is numpy this must be provided.  
        OUTPUT: 
            combinations - a dict where the key is a gene and the value is a list of 
                           all regualtory groups and their combined amplification (P)
        """
            
        # if network is pandas dataframe turn that in to a numpy along with gene names in a list.
        if isinstance(network, pd.DataFrame):
            # if names are not set seperatly assign the index as names 
            if not names:
                names = network.index.astype(str).tolist()
            # and set network to a numpy array
            network = network.values
        elif isinstance(network, np.ndarray) and not names:
            raise ValueError("When using numpy.array for combination generation genesnake.model.parameters the gene names must be provided in the names argument.")
        else:
            raise TypeError("Network input to genesnake.model.parameters.generateSP(combinations) must be of type numpy.ndarray or pandas.DataFrame.")
        
        # to make sure nothing strange happens check that names and rows are the same number
        if not len(names)==network.shape[0]:
            raise ValueError("The number of given names seems to be different from the number of genes in the network in genesnake.model.parameters.generateSP(combinations)")
        
        # make sure that combinations of length 1 is not created as this will cause issues later
        # on and does not make sense instead return false telling the other tools that no combinations are possible
        if max_comb < 2:
            return False
        
        # create an empty dict 
        combinations = dict() 
        
        # for each gene find the regulators of said gene 
        for i in range(network.shape[0]):
            # starting from combinations of 2 genes we move up until max_comb
            combs = 2
            
            # for each gene select from the list of names 
            combinations[names[i]] = list()
            # Find the positive and negative regulators
            # This is so that we do not draw a combinations of positive and
            # negative interactions as this would be impossible to explain
            # in the network.
            posinds = [names[j] for j in np.where(network[:,i]>0)[0]]
            neginds = [names[j] for j in np.where(network[:,i]<0)[0]]
            # then loop opver each set of interactions and make combinations of length combs
            # this is done until we reach combinations of max_comb
            while combs<=max_comb:
                # draw all possible combinations of upregulators in length combs
                combinations[names[i]] = combinations[names[i]]+["_".join(string) for string in list(itertools.combinations(posinds,combs))]
                # do the same for inhibition 
                combinations[names[i]] = combinations[names[i]]+["_".join(string) for string in list(itertools.combinations(neginds,combs))]
                # and add one more size of combinations 
                combs=combs+1
           
            # then we assign 1 or 0 to each possible combination to weight this in the model 
            for j in range(len(combinations[names[i]])):
                # draw a random float in range 0-1 
                P = np.random.rand()
                # then assign 1 if P is < the proability to collaborate 
                if P < comb_prob:
                    combinations[names[i]][j] = combinations[names[i]][j]+":"+str(1)
                # else assign 0 
                else:
                    combinations[names[i]][j] = combinations[names[i]][j]+":"+str(0)
                    
        return combinations
    
    
    def make_alphas(network,combinations,genes=None):
        # if network is pandas dataframe turn that in to a numpy along with gene names in a list.
        if isinstance(network, pd.DataFrame):
            # if names are not set seperatly assign the index as names 
            if not genes:
                genes = network.index.astype(str).tolist()
            # and set network to a numpy array
            network = network.values
        elif isinstance(network, np.ndarray) and not names:
            raise ValueError("When using numpy.array for generation of alphas in genesnake.model.parameters the gene names must be provided in the names argument.")
        else:
            raise TypeError("Network input to genesnake.model.parameters.generateSP(alphas) must be of type numpy.ndarray or pandas.DataFrame.")
        
        if not isinstance(combinations,dict):
            raise TypeError("combinations input to genesnake.model.parameters.generateSP(alphas) must be of type dict.")
        
        # assign some varaibles that we will need for all loops 
        alphas = dict()
        
        # get the median edge strength as an indicator if a regulation is strong or weak
        # so that alpha can be adjusted based on this 
        edge_median = np.median(np.abs(network[np.where(network)]))
        
        # set genes to string as combinations.keys() will be string objects
        genes = [str(g) for g in genes]
        
        # for each gene get the regulators 
        for i in range(len(genes)):
            # check if the gene is regulated by multiple other genes
            # and if so do any of them colaberate
            # if not no aplha is added 
            if len(combinations[genes[i]])>0:
                for comb in combinations[genes[i]]:
                    # if this is an AND relation we set
                    # the combination high and the 
                    if comb.split(":")[-1] == '1':
                        # get the genes involved in the combo
                        regs = comb.split(":")[0].split("_")
                        inds = list()
                        # get the strenght for each regulator in the network
                        for reg in regs:
                            inds.append(genes.index(reg))
                        # add these to a list and turn absolute
                        reg_strength = np.abs(network[inds,i])
                        # then check if this is a strong or weak regulation 
                        if any(reg_strength)>=edge_median:
                            # if strong assing a high value to the alpha 
                            alphas[comb+"_"+genes[i]] = np.round(np.random.uniform(0.7,1),2)
                            # and a weak value to each regulator+target without the AND 
                            for reg in regs:
                                if reg+"_"+genes[i] not in alphas.keys():
                                    alphas[reg+"_"+genes[i]] = np.round(np.random.uniform(0.2,0.4),2)
                        else:
                            # else asign a weak regulation to both the AND and each seperate regulator+target
                            alphas[comb+"_"+genes[i]] = np.round(np.random.uniform(0.5,0.6),2)
                            for reg in regs:
                                if reg+"_"+genes[i] not in alphas.keys():
                                    alphas[reg+"_"+genes[i]] = np.round(np.random.uniform(0.2,0.4),2)
            
            # once all combinatory relations are solved add any remaning alphas
            inds = np.where(network[:,i])[0]
            for j in inds:
                reg = genes[j]
                # if the alpha exist do not resign it 
                if reg+"_"+genes[i] not in alphas.keys():
                    # if there is a strong regulator assing a high alpha 
                    if np.abs(network[j,i])>edge_median:
                        alphas[reg+"_"+genes[i]] = np.round(np.random.uniform(0.6,1),2)
                    # otherwise assing a semi strong regulator as the method for determening
                    # regultor strenght is rather poor
                    else:
                        alphas[reg+"_"+genes[i]] = np.round(np.random.uniform(0.4,0.8),2)
        
        # and return the dict 
        return(alphas)
    
    
    def make_dis_const(genes=None,Emax=0.6,Emin=0.2):
        if genes is None:
            raise ValueError("Attempted to generate parameters for None object in genesnake.modle.parameters")
        return np.round(np.random.uniform(Emin,Emax,(len(genes))),2).tolist()
    
    
    def make_lambda_RNA(genes=None,Emax=0.7,Emin=0.5):
        if genes is None:
            raise ValueError("Attempted to generate parameters for None object in genesnake.modle.parameters")
        return np.round(np.random.uniform(Emin,Emax,(len(genes))),2).tolist()
    
    
    def make_lambda_prot(genes=None,Emax=0.4,Emin=0.2):
        if genes is None:
            raise ValueError("Attempted to generate parameters for None object in genesnake.modle.parameters")
        return np.round(np.random.uniform(Emin,Emax,(len(genes))),2).tolist()
