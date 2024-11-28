"""
A collection of functions that are used to randomly generate parameters for the modle.py script.
Note that they are not imported in to the data namespace only the model class. 
"""

from scipy.stats import truncnorm
import pandas as pd
import numpy as np
import itertools


def get_truncated_normal(low, up, mean=0, sd=1):
    """
    Function for wrapping up truncnorm in a more handy function. 
    Should generally only be used inside the perturbation functions to 
    obtain noisy perturbations centered around a given mean.
    """
    # draw a value from a continues truncated normal distrubution,
    # a normal distrubution with tails that are stoped at min/max value 
    val = truncnorm((low - mean) / sd, (up - mean) / sd, loc=mean, scale=sd)
    # return the generated float value only 
    return round(val.rvs(),2)


def get_noise(noise):
    """
    simple function that returns +/- noise based on a 50% random 
    """
    if noise == 0:
        return(0)
    
    r = np.random.random()
    
    noise = get_truncated_normal(0,noise)
    
    if r > 0.5:
        return noise
    else:
        return noise*-1


def fix_perts(pert_mat,sign):
    """
    Function for ensuring that noise cannot change perturbations to be less than 0 or more than -100% depending for -1 sign
    """
    if sign == -1:
        pert_mat[np.where(pert_mat<-1.0)] = -1.0
        pert_mat[np.where(pert_mat>0.0)] = 0.0
    if sign == 1:
        pert_mat[np.where(pert_mat<0.0)] = 0.0
        
    return pert_mat


class pertgen:
    """
    Class for simplyfying calling a set of perturbation schemes. 
    Can with advantage be used via genesnake.ODEmodel.get_pert()
    Is not imported in the genesnake namespace but could be manually imported in principle. 
    """
    
    def diagonal(genes,effect,reps,noise,sign):
        """
        Function that creates a simple diagonal perturbation matrix with either up or down regulation of gene expression. 
        INPUT:
            genes - the names of the genes for which to create a perturbation on, is also used to determine the size of the output matrix.
                    Will be automatically assigned when using genesnake.ODEmodel.get_pert().
            effect - How strong the perturbation effect is. Can be a float between 1-0 or a tuple with two floats between 1-0. 
                     1 meaning no perturbation and 0 being a complete knockout.
            reps - how many identical experiments should be generated as replicates. 
            exp - how many differing experiemnts should be generated (act as additional reps for diagonal pert)
            noise - a float of how much the effect can vary from the intended effect. E.g. if effect = 0.7 and noise = 0.1 perturbation
                    effect will range from 0.6-0.8 centering on 0.7.
                    Note: This can not change sign of the perturbation (down<=1,up>=1) nor can it cause negative expression (>0). 
            sign - either \"up\" or \"down\". determines the direction of the perturbation up for over expression and down for knockdowns.
        
        OUTPUT: 
            pert_mat - a pandas dataframe of size genesXgenes*reps with a perturbation of +/-0.5 of each gene along the diagonal
        """
        
        # get the number of genes in the system 
        N = len(genes)
        
        # set a single replicate as a start 
        pert_mat=np.eye(N)
        columns = ["E"+str(g+1)+"R1" for g in range(len(genes))]
        
        # add a diagonal matrix for all other replicates 
        for i in range(reps-1):
            pert_mat = np.hstack([pert_mat,np.eye(N)])
            columns = columns + ["E"+str(g+1)+"R"+str(i+2) for g in range(len(genes))]
                        
        # clean up so that all 0s are positive, this does not matter from a math standpoint but
        # looks slightly better with 0 than -0
        pert_mat[np.nonzero(pert_mat==0)] = 0
        
        # add weights to the system
        if isinstance(effect,tuple):
            for i in range(reps*N):
                pert_mat[:,i] = pert_mat[:,i]*(get_truncated_normal(effect[0],effect[1],round(np.mean(effect),2))+get_noise(noise))
        else:
            for i in range(reps*N):
                pert_mat[:,i] = pert_mat[:,i]*(effect+get_noise(noise))
        
        # if negative sign make values negative 
        if sign<0:
            pert_mat[np.nonzero(pert_mat)] = -1*pert_mat[np.nonzero(pert_mat)]
        
        # make sure that the sign is consistent and that there are no more than 100% reduction in expression
        pert_mat = fix_perts(pert_mat,sign)
            
        return pd.DataFrame(pert_mat,index=genes,columns=columns)
    
    
    def multiple_pert(genes,effect,reps,exp,noise,sign,multipert):
        """
        Function that creates a perturbation matrix with multiple genes perturbed in each experiment. 
        It can be either up or down regulation of gene expression. 
        INPUT:
            genes - the names of the genes for which to create a perturbation on, is also used to determine the size of the output matrix.
                    Will be automatically assigned when using genesnake.ODEmodel.get_pert().
            effect - How strong the perturbation effect is. Can be a float between 1-0 or a tuple with two floats between 1-0. 
                     1 meaning no perturbation and 0 being a complete knockout.
            reps - how many identical experiments should be generated as replicates. 
            exp - how many differing experiemnts should be generated (act as additional reps for diagonal pert)
            noise - a float of how much the effect can vary from the intended effect. E.g. if effect = 0.7 and noise = 0.1 perturbation
                    effect will range from 0.6-0.8 centering on 0.7.
                    Note: This can not change sign of the perturbation (down<=1,up>=1) nor can it cause negative expression (>0). 
            sign - either \"up\" or \"down\". determines the direction of the perturbation up for over expression and down for knockdowns.
            multipert - int>2 the number of perturbations per experiment. 
        
        OUTPUT: 
            pert_mat - a pandas dataframe of size genesXgenes*reps with a perturbation of +/-0.5 of each gene along the diagonal
        """
        
        # get the number of genes in the system 
        N = len(genes)
        
        # make a numpy array of size N,exp
        pert_mat = np.zeros((N,exp))
        
        # form the zero array we start randomly adding pertrubations until each column has multipert perts
        # in a way so that each gene should be pertrubed at least once when possible
        for i in range(pert_mat.shape[1]):
            while np.count_nonzero(pert_mat[:,i]) < multipert:
                # get the rowsum for each row. This is used to ensure an even spread
                # among which genes are pertrubed 
                rowsum = np.sum(pert_mat,axis=1)
                # get a random vector to make sure that the pertrubations are semi random distrubuted 
                rands = np.random.rand(N)
                # add the two vectors up 
                rs = rowsum+rands
                # and pick the smallest value so that each gene is roughly equally perturbed 
                pos = np.argmin(rs)
                
                # finally add a weight to the perturbation either from a truncated normal distrubution or from the effect value 
                if isinstance(effect,tuple):
                    pert_mat[pos,i] = get_truncated_normal(effect[0],effect[1],round(np.mean(effect),2))+get_noise(noise)
                else:
                    pert_mat[pos,i] = effect+get_noise(noise)
        
        # once all experiemnts are created expand them with the number of replicates wanted 
        pert_mat = np.repeat(pert_mat,reps,axis=1)
        
        # create a list of colum names in format E1R1 to represent experiment (E) and replicate (R)
        columns = list()
        for i in range(exp):
            string = "E"+str(i+1)
            for j in range(reps):
                columns.append(string+"R"+str(j+1))
        
        # if negative sign make values negative 
        if sign<0:
            pert_mat[np.nonzero(pert_mat)] = -1*pert_mat[np.nonzero(pert_mat)]
        
        # make sure that the sign is consistent and that there are no more than 100% reduction in expression
        pert_mat = fix_perts(pert_mat,sign)
        
        return pd.DataFrame(pert_mat,index=genes,columns=columns)
  
    
    def random_pert(genes,effect,reps,exp,noise,sign,multipert):
        """
        Function that creates a perturbation matrix with multiple genes perturbed in each experiment. 
        It can be either up or down regulation of gene expression. 
        INPUT:
            genes - the names of the genes for which to create a perturbation on, is also used to determine the size of the output matrix.
                    Will be automatically assigned when using genesnake.ODEmodel.get_pert().
            effect - How strong the perturbation effect is. Can be a float between 1-0 or a tuple with two floats between 1-0. 
                     1 meaning no perturbation and 0 being a complete knockout.
            reps - how many identical experiments should be generated as replicates. 
            exp - how many differing experiemnts should be generated (act as additional reps for diagonal pert)
            noise - a float of how much the effect can vary from the intended effect. E.g. if effect = 0.7 and noise = 0.1 perturbation
                    effect will range from 0.6-0.8 centering on 0.7.
                    Note: This can not change sign of the perturbation (down<=1,up>=1) nor can it cause negative expression (>0). 
            sign - either \"up\" or \"down\". determines the direction of the perturbation up for over expression and down for knockdowns.
            multipert - int>2 the number of perturbations per experiment. 
        
        OUTPUT: 
            pert_mat - a pandas dataframe of size genesXgenes*reps with a perturbation of +/-0.5 of each gene along the diagonal
        """
        
        # get the number of genes in the system 
        N = len(genes)
        
        # make a numpy array of size N,exp
        pert_mat = np.zeros((N,exp))
        
        # form the zero array we start randomly adding pertrubations until each column has multipert perts
        for i in range(pert_mat.shape[1]):
            while np.count_nonzero(pert_mat[:,i]) < multipert:
                # create a random vector to select where to add a pertrubation
                rs = np.random.rand(N)
                pos = np.argmin(rs)
                
                # and add weight to it either from a truncated normal distrubution or the selected effect
                if isinstance(effect,tuple):
                    pert_mat[pos,i] = get_truncated_normal(effect[0],effect[1],round(np.mean(effect),2))+get_noise(noise)
                else:
                    pert_mat[pos,i] = effect+get_noise(noise)
        
        # once all experiemts are done create corresponding replicates
        pert_mat = np.repeat(pert_mat,reps,axis=1)
        
        # create a list of colum names in format E1R1 to represent experiment (E) and replicate (R)
        columns = list()
        for i in range(exp):
            string = "E"+str(i+1)
            for j in range(reps):
                columns.append(string+"R"+str(j+1))
            
        # if negative sign make values negative 
        if sign<0:
            pert_mat[np.nonzero(pert_mat)] = -1*pert_mat[np.nonzero(pert_mat)]
        
        # make sure that the sign is consistent and that there are no more than 100% reduction in expression
        pert_mat = fix_perts(pert_mat,sign)
        
        return pd.DataFrame(pert_mat,index=genes,columns=columns)
    
    
    def all_pert(genes,effect,reps,exp,noise,sign):
        """
        Function for perturbing all genes in an experiment. It is recommended to only use small pertbations here but it is not limited. 
        It is also recommended to use a range of effect for this one but if not it will use +/-effect to simulate this differece. 
        INPUT:
            genes - the names of the genes for which to create a perturbation on, is also used to determine the size of the output matrix.
                    Will be automatically assigned when using genesnake.ODEmodel.get_pert().
            effect - How strong the perturbation effect is. Can be a float between 1-0 or a tuple with two floats between 1-0. 
                     1 meaning no perturbation and 0 being a complete knockout.
            reps - how many identical experiments should be generated as replicates. 
            exp - how many differing experiemnts should be generated (act as additional reps for diagonal pert)
            noise - a float of how much the effect can vary from the intended effect. E.g. if effect = 0.7 and noise = 0.1 perturbation
                    effect will range from 0.6-0.8 centering on 0.7.
                    Note: This can not change sign of the perturbation (down<=1,up>=1) nor can it cause negative expression (>0).        
        OUTPUT: 
            pert_mat - a pandas dataframe of size genesXgenes*reps with a perturbation of +/-0.5 of each gene along the diagonal
        """
        
        # get the number of genes in the system 
        N = len(genes)
        
        # if effect is a single float make a narrow span of +/-10% of inital value
        if isinstance(effect,float):
            effect = (effect-effect*0.1,effect+effect*0.1)
        
        # draw a set of random values in range of effect 
        val = truncnorm.rvs((effect[0]-0)/1.0,(effect[1]-0) / 1, loc=0, scale=1,size=(N,exp))
        # add a random sign to these perturbations
        sign = 2*np.random.randint(0,2,size=(N,exp))-1
        val = np.multiply(val,sign)
        # get a noise matrix for this
        noise = truncnorm.rvs((0-0)/1.0,(noise-0) / 1, loc=0, scale=1,size=(N,exp))
        nosie = np.multiply(noise,(2*np.random.randint(0,2,size=(N,exp))-1))
        # and finally convert the perturbation to percentage effect from 100% 
        pert_mat = val+noise
        
        # make sure no values are less than zero
        pert_mat[np.where(pert_mat<0)] = 0.0
        
        # once all experiemts are done create corresponding replicates
        pert_mat = np.repeat(pert_mat,reps,axis=1)
        
        # create a list of colum names in format E1R1 to represent experiment (E) and replicate (R)
        columns = list()
        for i in range(exp):
            string = "E"+str(i+1)
            for j in range(reps):
                columns.append(string+"R"+str(j+1))
        
        return pd.DataFrame(pert_mat,index=genes,columns=columns)
    
    
    def combined_pert(genes,net,effect,reps,exp,noise,sign,multipert):
        """
        Function that creates a perturbation matrix with multiple genes perturbed in each experiment. 
        It can be either up or down regulation of gene expression. 
        INPUT:
            genes - the names of the genes for which to create a perturbation on, is also used to determine the size of the output matrix.
                    Will be automatically assigned when using genesnake.ODEmodel.set_pert().
            net - the network that combinations are selected from. Will be automatically assigned when using 
                  genesnake.ODEmodel.set_pert().
            effect - How strong the perturbation effect is. Can be a float between 1-0 or a tuple with two floats between 1-0. 
                     1 meaning no perturbation and 0 being a complete knockout.
            reps - how many identical experiments should be generated as replicates. 
            exp - how many differing experiemnts should be generated (act as additional reps for diagonal pert)
            noise - a float of how much the effect can vary from the intended effect. E.g. if effect = 0.7 and noise = 0.1 perturbation
                    effect will range from 0.6-0.8 centering on 0.7.
                    Note: This can not change sign of the perturbation (down<=1,up>=1) nor can it cause negative expression (>0). 
            sign - either \"up\" or \"down\". determines the direction of the perturbation up for over expression and down for knockdowns.
            multipert - int>2 the number of perturbations per experiment. 
        
        OUTPUT: 
            pert_mat - a pandas dataframe of size genesXgenes*reps with a perturbation of +/-0.5 of each gene along the diagonal
        """
        
        # get the number of genes in the system 
        N = len(genes)
        
        
        # get all possible combinations of the regulators of size multipert
        combs = list()
        for i in range(N):
            # if there are multipert or more possible combinations create all possible multipert combinations
            # note that if there is only multipert regulators there is only 1 possible combination
            if np.count_nonzero(net.values[:,i]) >= multipert:
                regulates = net.index[np.nonzero(net.values[:,i])].tolist()
                combs = combs+list(itertools.combinations(regulates,multipert))
                
        # remove experiemnts that would be replicates, these can arrise if multiple genes are regulated by the same set
        combs = list(set(combs))
        
        # make sure there are some combinations to work with
        if len(combs) == 0:
            raise GeneratorExit("Failed to find any possible combinations of length "+str(multipert)+" in genesnake.perturbation_generators.combined_pert.")
        
        # make sure that there are at exp combinatsion
        if len(combs)<exp:
            print("WARNING: Could not find enough combinations to fill all experiemnts in genesnake.perturbation_generators.combined_pert.")
            print("Will create a perturbation matrix with only "+str(len(combs))+" experiments instead the requested "+str(exp))
            exp = len(combs)
        
        # make a numpy array of size N,exp
        pert_mat = np.zeros((N,exp))
        
        # once all possible combinations of size 2-multipert have been found
        # randomly select from these until all experiemnts have been filled 
        for i in range(pert_mat.shape[1]):
            # select the pertrubations to do 
            select = np.random.randint(len(combs))
            experiment = combs[select]
            # and drop this from possible further experiemnts
            combs.remove(combs[select])
            for j in experiment:
                pos = genes.index(j)
                # finally add a weight to the perturbation either from a truncated normal distrubution or from the effect value 
                if isinstance(effect,tuple):
                    pert_mat[pos,i] = get_truncated_normal(effect[0],effect[1],round(np.mean(effect),2))+get_noise(noise)
                else:
                    pert_mat[pos,i] = effect+get_noise(noise)
        
        # once all experiemnts are created expand them with the number of replicates wanted 
        pert_mat = np.repeat(pert_mat,reps,axis=1)
        
        # create a list of colum names in format E1R1 to represent experiment (E) and replicate (R)
        columns = list()
        for i in range(exp):
            string = "E"+str(i+1)
            for j in range(reps):
                columns.append(string+"R"+str(j+1))
        
        # if negative sign make values negative 
        if sign<0:
            pert_mat[np.nonzero(pert_mat)] = -1*pert_mat[np.nonzero(pert_mat)]
        
        # make sure that the sign is consistent and that there are no more than 100% reduction in expression
        pert_mat = fix_perts(pert_mat,sign)
        
        return pd.DataFrame(pert_mat,index=genes,columns=columns)
    
    
    def targeted_pert(genes,effect,reps,noise,sign,targets):
        """
        Function that creates a perturbation matrix with multiple genes perturbed in each experiment. 
        It can be either up or down regulation of gene expression. 
        INPUT:
            genes - the names of the genes for which to create a perturbation on, is also used to determine the size of the output matrix.
                    Will be automatically assigned when using genesnake.ODEmodel.set_pert().
            net - the network that combinations are selected from. Will be automatically assigned when using 
                  genesnake.ODEmodel.set_pert().
            effect - How strong the perturbation effect is. Can be a float between 1-0 or a tuple with two floats between 1-0. 
                     1 meaning no perturbation and 0 being a complete knockout.
            reps - how many identical experiments should be generated as replicates. 
            noise - a float of how much the effect can vary from the intended effect. E.g. if effect = 0.7 and noise = 0.1 perturbation
                    effect will range from 0.6-0.8 centering on 0.7.
                    Note: This can not change sign of the perturbation (down<=1,up>=1) nor can it cause negative expression (>0). 
            sign - either \"up\" or \"down\". determines the direction of the perturbation up for over expression and down for knockdowns.
            targets - List of gene names to be perturbed. Used in pert scheme specific to determine which genes will be perturbed. 
                      NOTE to include multiple pertbations in a single experiemnt use a list of lists e.g. [name1,name2,[name2,name1]..]
            
        OUTPUT: 
            pert_mat - a pandas dataframe of size genesXgenes*reps with a perturbation of +/-0.5 of each gene along the diagonal
        """
        
        # get the number of genes in the system 
        N = len(genes)
        
        # make a numpy array of size N,number of targets
        pert_mat = np.zeros((N,len(targets)))
        
        # form the zero array we start randomly adding pertrubations until each column has multipert perts
        for i in range(pert_mat.shape[1]):
            if isinstance(targets[i],list):
                for j in targets[i]:
                    pos = genes.index(j)
                    # and add weight to it either from a truncated normal distrubution or the selected effect
                    if isinstance(effect,tuple):
                        pert_mat[pos,i] = get_truncated_normal(effect[0],effect[1],round(np.mean(effect),2))+get_noise(noise)
                    else:
                        pert_mat[pos,i] = effect+get_noise(noise)
            else:
                pos = genes.index(targets[i])
                # and add weight to it either from a truncated normal distrubution or the selected effect
                if isinstance(effect,tuple):
                    pert_mat[pos,i] = get_truncated_normal(effect[0],effect[1],round(np.mean(effect),2))+get_noise(noise)
                else:
                    pert_mat[pos,i] = effect+get_noise(noise)
        
        # once all experiemts are done create corresponding replicates
        pert_mat = np.repeat(pert_mat,reps,axis=1)
        
        # create a list of colum names in format E1R1 to represent experiment (E) and replicate (R)
        columns = list()
        for i in range(len(targets)):
            string = "E"+str(i+1)
            for j in range(reps):
                columns.append(string+"R"+str(j+1))
        
        # if negative sign make values negative 
        if sign<0:
            pert_mat[np.nonzero(pert_mat)] = -1*pert_mat[np.nonzero(pert_mat)]
            
        # make sure that the sign is consistent and that there are no more than 100% reduction in expression
        pert_mat = fix_perts(pert_mat,sign)
        
        return pd.DataFrame(pert_mat,index=genes,columns=columns)
    
#!# should a get_pert be added here to return a seperate P matrix without using the ODEmodel class?
