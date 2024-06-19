"""
A collection of functions that are used to randomly generate parameters for the modle.py script.
Note that they are not imported in to the data namespace only the model class. 
"""

import scipy.stats as stats
import pandas as pd
import numpy as np
import itertools

class pargen:
    
    def make_start_rna(genes=None,high_exp=10,low_exp=5):
        if genes is None:
            raise ValueError("Attempted to generate parameters for None object in genesnake.modle.parameters")
                
        mean = (high_exp+low_exp)/2
        std = 1
        
        RNA = list()
        for i in range(len(genes)):
            RNA.append(np.round(stats.truncnorm.rvs((low_exp-mean)/std,(high_exp-mean)/std,loc=mean,scale=std),2))
            
        #A= np.round(np.random.uniform(low=low_exp,high=high_exp,size=(len(genes))),2).tolist()
        return RNA
    
    def make_start_prot(genes=None,high_exp=10,low_exp=5):
        if genes is None:
            raise ValueError("Attempted to generate parameters for None object in genesnake.modle.parameters")
        
        mean = (high_exp+low_exp)/2
        std = 1
        
        prot = list()
        for i in range(len(genes)):
            prot.append(np.round(stats.truncnorm.rvs((low_exp-mean)/std,(high_exp-mean)/std,loc=mean,scale=std),2))
            
        #A= np.round(np.random.uniform(low=low_exp,high=high_exp,size=(len(genes))),2).tolist()
        return prot
        
    
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
    
    
    def make_hill_coef(genes=None,Emax=4,Emin=2):#3-5
        if genes is None:
            raise ValueError("Attempted to generate parameters for None object in genesnake.modle.parameters")
        
        return np.round(np.random.randint(low=Emin,high=Emax,size=(len(genes))),2).tolist()
    
    
    def make_combinations(network,names=None,max_comb=2,comb_prob=0.5):
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
            inds = np.where(network[:,i]!=0)[0]
            # then loop opver each set of interactions and make combinations of length combs
            # this is done until we reach combinations of max_comb
            while combs<=len(inds):
                new_comb = None
                for string in list(itertools.combinations(inds,combs)):
                    if all([val>0 for val in network[string,i]]) or all([val<0 for val in network[string,i]]):
                        add = [names[j] for j in string]
                        new_comb = "_".join(add)
                        # draw all possible combinations of upregulators in length combs
                        combinations[names[i]].append(new_comb)
                # and add one more size of combinations 
                combs=combs+1

            # then assign a 1 or 0 to collaboration constant 
            for j in range(len(combinations[names[i]])):
                if len(combinations[names[i]][j].split("_")) > max_comb:
                    combinations[names[i]][j] = combinations[names[i]][j]+":"+str(0)
                else:
                    r = np.random.random()
                    if r < comb_prob:
                        combinations[names[i]][j] = combinations[names[i]][j]+":"+str(1)
                    else:
                        combinations[names[i]][j] = combinations[names[i]][j]+":"+str(0)
        return combinations
    
    
    def make_alphas(network,combinations,genes=None,maxa=1.0,mina=0.9):
        """
        Function for generating alpha values which controls how strong effect a regulator has on it's targets.
        INPUT:
            - network - the network which determines regulator target relationship
            - combinations - a dictonary that contains information on what regulators have a combinatory effect and thus need an alhpa for this
            - genes - the names to use for the generated alpha values. Only needed if you for some reason use different names for the network and the model.
            - maxa - the maximum float value alpha can take.
            - mina - the minimum float value alpha can take.
        OUTPUT:
            - alphas - dictonary which contains the alpha for each regulator per target
        """
        # if network is pandas dataframe turn that in to a numpy along with gene names in a list.
        if isinstance(network, pd.DataFrame):
            # if names are not set seperatly assign the index as names 
            if not genes:
                genes = network.index.astype(str).tolist()
            # and set network to a numpy array
            network = network.values
        elif isinstance(network, np.ndarray) and not genes:
            raise ValueError("When using numpy.array for generation of alphas in genesnake.model.parameters the gene names must be provided in the genes argument.")
        else:
            raise TypeError("Network input to genesnake.model.parameters.generateSP(alphas) must be of type numpy.ndarray or pandas.DataFrame.")
        
        if not isinstance(combinations,dict):
            raise TypeError("combinations input to genesnake.model.parameters.generateSP(alphas) must be of type dict.")
        
        # assign some varaibles that we will need for all loops 
        alphas = dict()
        
        
        # set genes to string as combinations.keys() will be string objects
        genes = [str(g) for g in genes]
        
        # for each gene get the regulators 
        for i in range(len(genes)):
            # first add the passive expression, this is a random value between min_self and max_self
            # define these here to make it easier to change 
            min_self = 0.01
            max_self = 0.015
            mean = (min_self+max_self)/2
            std = 0.1
            alphas[genes[i]] = np.round(stats.truncnorm.rvs((min_self-mean)/std,(max_self-mean)/std,loc=mean,scale=std),2)
            
            mean = (maxa+mina)/2
            std = 1
            
            # check if the gene is regulated by multiple other genes
            # and if so do any of them colaberate
            # if not no aplha is added 
            if len(combinations[genes[i]])>0:
                for comb in combinations[genes[i]]:
                    # if this is an AND relation we set
                    # the combination high and the 
                    if float(comb.split(":")[-1]) != 0:
                        # get the genes involved in the combo
                        regs = comb.split(":")[0].split("_")
                        inds = list()
                        # get the strenght for each regulator in the network
                        for reg in regs:
                            inds.append(genes.index(reg))
                        
                        if any(val<0 for val in network[inds,i]):
                            alphas[genes[i]+"_"+comb.split(":")[0]] = 0
                            
                            # finally map a weaker value for each of the combined regulators
                            # it is arbitrarily assigned a value 10x smaller than the mapped value to
                            # maintain some minimal effect on the system
                            val = np.round(stats.truncnorm.rvs((mina-mean)/std,(maxa-mean)/std,loc=mean,scale=std),2)                   
                            for reg in regs:
                                reg_strength = network[genes.index(reg),i]
                                if reg_strength < 0:
                                    alphas[genes[i]+"_"+reg] = 0
                                else:
                                    alphas[genes[i]+"_"+reg] = val
                        else:
                            alphas[genes[i]+"_"+comb.split(":")[0]] = np.round(stats.truncnorm.rvs((mina-mean)/std,(maxa-mean)/std,loc=mean,scale=std),2)
                            
                            new_mina = mina*0.1
                            new_maxa = maxa*0.1
                            new_mean = (new_maxa+new_mina)/2
                            new_std = 0.1
                            val = np.round(stats.truncnorm.rvs((new_mina-new_mean)/std,(new_maxa-new_mean)/new_std,loc=new_mean,scale=std),2)
                            
                            for reg in regs:
                                reg_strength = network[genes.index(reg),i]
                                if reg_strength < 0:
                                    alphas[genes[i]+"_"+reg] = 0
                                else:
                                    new_mina = mina*0.1
                                    new_maxa = maxa*0.1
                                    new_mean = (new_maxa+new_mina)/2
                                    new_std = 0.1
                                    alphas[genes[i]+"_"+reg] = val
                    else:
                        regs = comb.split(":")[0].split("_")
                        inds = list()
                        # get the strenght for each regulator in the network
                        for reg in regs:
                            inds.append(genes.index(reg))
                            
                        val = np.round(stats.truncnorm.rvs((mina-mean)/std,(maxa-mean)/std,loc=mean,scale=std),2)                   
                        for reg in regs:
                            reg_strength = network[genes.index(reg),i]
                            if reg_strength < 0:
                                alphas[genes[i]+"_"+reg] = 0
                            else:
                                alphas[genes[i]+"_"+reg] = val
                            
            # once all combinatory relations are solved add any remaning alphas
            inds = np.where(network[:,i])[0]
            for j in inds:
                reg = genes[j]
                # if the alpha exist do not resign it 
                if genes[i]+"_"+reg not in alphas.keys():
                    # map the regulator to the weight in the network
                    if network[j,i] < 0:
                        alphas[genes[i]+"_"+reg] = 0
                    else:
                        alphas[genes[i]+"_"+reg] = np.round(stats.truncnorm.rvs((mina-mean)/std,(maxa-mean)/std,loc=mean,scale=std),2)
                                                            
        # and return the dict 
        return(alphas)
    
    
    def make_dis_const(network,combinations,genes=None,Emax=0.8,Emin=0.5):
        """
        Function for generating the disociation constant (k) values which controls regulation strenght by modeling a molecule releasing rather then regulating.
        INPUT:
            - network - the network which determines regulator target relationship
            - combinations - a dictonary that contains information on what regulators have a combinatory effect and thus need an alhpa for this
            - genes - the names to use for the generated alpha values. Only needed if you for some reason use different names for the network and the model.
            - Emax - the maximum float value k can take.
            - Emin - the minimum float value k can take.
        OUTPUT:
            - dis_const - dictonary which contains the k values for each regulator per target
        """

        # if network is pandas dataframe turn that in to a numpy along with gene names in a list.
        if isinstance(network, pd.DataFrame):
            # if names are not set seperatly assign the index as names 
            if not genes:
                genes = network.index.astype(str).tolist()
            # and set network to a numpy array
            network = network.values
        elif isinstance(network, np.ndarray) and not genes:
            raise ValueError("When using numpy.array for generation of alphas in genesnake.model.parameters the gene names must be provided in the genes argument.")
        else:
            raise TypeError("Network input to genesnake.model.parameters.generateSP(alphas) must be of type numpy.ndarray or pandas.DataFrame.")
        
        if not isinstance(combinations,dict):
            raise TypeError("combinations input to genesnake.model.parameters.generateSP(alphas) must be of type dict.")

        # get mean
        mean = (Emax+Emin)/2
        # and simply assign a random std
        std = 1
        
        # assign some varaibles that we will need for all loops 
        dis = dict()
            
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
                    if float(comb.split(":")[-1]) != 0:
                        dis[genes[i]+"_"+comb.split(":")[0]] = np.round(stats.truncnorm.rvs((Emin-mean)/std,(Emax-mean)/std,loc=mean,scale=std),2)
            # once all combinatory relations are solved add any remaning alphas
            inds = np.where(network[:,i])[0]
            for j in inds:
                reg = genes[j]
                # if the alpha exist do not resign it 
                if genes[i]+"_"+reg not in dis.keys():
                    # map the regulator to the weight in the network
                    dis[genes[i]+"_"+reg] = np.round(stats.truncnorm.rvs((Emin-mean)/std,(Emax-mean)/std,loc=mean,scale=std),2)
                
        return dis
    
    
    def make_lambda_RNA(genes=None,Emax=0.8,Emin=0.7):
        """
        Based on the GNW function for getting the degradation. 
        Emax and Emin is halftime for protein in minutes 
        """

        if genes is None:
            raise ValueError("Attempted to generate parameters for None object in genesnake.modle.parameters")
        
        # get mean
        mean = (Emax+Emin)/2
        # and simply assign a random std
        std = 1
        
        deltas = list()
        for i in range(len(genes)): 
            deltas.append(np.round(stats.truncnorm.rvs((Emin-mean)/std,(Emax-mean)/std,loc=mean,scale=std),2))
        
        return deltas
    
    
    def make_lambda_prot(genes=None,Emax=0.8,Emin=0.7):
        """
        Based on the GNW function for getting the degradation. 
        Emax and Emin is halftime for protein in minutes 
        """

        if genes is None:
            raise ValueError("Attempted to generate parameters for None object in genesnake.modle.parameters")
        if genes is None:
            raise ValueError("Attempted to generate parameters for None object in genesnake.modle.parameters")
        
        # get mean
        mean = (Emax+Emin)/2
        # and simply assign a random std
        std = 1
        
        deltas = list()
        for i in range(len(genes)): 
            deltas.append(np.round(stats.truncnorm.rvs((Emin-mean)/std,(Emax-mean)/std,loc=mean,scale=std),2))
        
        return deltas
        
