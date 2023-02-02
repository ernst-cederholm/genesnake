# class for storing a geneSnoake model as a string object and all the parameters needed to use it
# created: 2022-11-08

from .perturbation_generators import pertgen
from .make_new_ODEmodel import make_new_ODEmodel
from .parameter_generators import pargen
from scipy.integrate import odeint
import scipy.stats as stats
import pandas as pd
import numpy as np
import itertools
import pickle
import types
import json
import sys


class ODEmodel(object):
    """
    geneSnake class for storing a ODE model as a string object along with all the nessesary parameters to run the model. 
    The model class further contains methods to run a model and to generate parameters in and model if needed. 
    """
    
    
    def __init__(self,model=None,parameters=None,network=None,genes=None):
        """
        Function for storing and initilization of a modle with corresponding parameters. 
        INPUT: 
            model - the source of the model for this object. Input can be: 
                    - None - an empty model class will be created
                    - file.txt - a txt file containg a ODE modle. The file will be read and stored in model.model
                    - str_object - a ODE modle in the form of a string object. It will be stored in model.model. 
        """
        self.steady_state_RNA = None
        self.steady_state_prot = None    
        self.ODEmodel = None    
        self.parameters = self.param(self)
        self.network = None
        self.genes = list()
        self._runing_ODEmodel = None
        self.perturbation = None
    
    def make_ODEmodel(network,*args,**kwargs):
        """
        add a bunch of documentation here
        """
        
        if isinstance(network,ODEmodel):
            self = network
            
            if self.network:
                network = self.network
            elif isinstance(args[0], pd.DataFrame) or isinstance(args[0], np.ndarray):
                network = args[0]
            else:
                raise ValueError("Could not find network to build ODE model from in genesnake.GRNmodel.make_model. Please add a network in pandas.DataFrame pr numpy.array to either the model.network argument or provide a network as the first argument on the function call")
            
            i = 1
        
            if self.genes:
                names = self.genes
            elif isinstance(args[i],list):
                names = args[i]
                i+=1
            else:
                names = None
                
            if len(self.parameters.combinations)>0:
                combines = self.parameters.combinations
            elif isinstance(args[i],dict) or args[i]==False:
                combines = args[i]
            elif "combinations" in kwargs.keys():
                combines = kwargs["combinations"] 
            else:
                if "max_comb" in kwargs.keys():
                    max_comb = kwargs[max_comb]
                else:
                    max_comb = 2
                if "comb_prob" in kwargs.keys():
                    comb_prob = kwargs[comb_prob]
                else:
                    comb_prob = 0.3
                                   
                combines = self.parameters.generateSP("combinations",network=network,max_comb=max_comb,comb_prob=comb_prob,names=names)
            
            self.ODEmodel= make_new_ODEmodel(network,combines,names)
        
        elif isinstance(network, pd.DataFrame) or isinstance(network, np.ndarray):
            network = network
            i = 0
            if "genes" in kwargs.keys():
                names = kwargs["genes"]
            elif isinstance(network, pd.DataFrame):
                names=list(network.columns)
            elif i < len(args):
                if isinstance(args[i],list):
                    names = args[i]
                    i+=1
            else:
                names = None
                
            if i < len(args):
                if isinstance(args[i],dict) or args[i]==False:
                    combines = args[i]
            elif "combinations" in kwargs.keys():
                combines = kwargs["combinations"] 
            else:
                if "max_comb" in kwargs.keys():
                    max_comb = kwargs[max_comb]
                else:
                    max_comb = 2
                    if "comb_prob" in kwargs.keys():
                        comb_prob = kwargs[comb_prob]
                    else:
                        comb_prob = 0.3
                            
                combines = pargen.make_combinations(network,max_comb=max_comb,comb_prob=comb_prob,names=names)
            
            # start a model class and return that.
            savemodel = ODEmodel.__new__(ODEmodel)
            # start a super class of the class 
            super(ODEmodel,savemodel).__init__()
            # and then start the class it self
            # a bit odd but requiered to make python behave. 
            savemodel.__init__(savemodel)
            # add all attributes that are known 
            savemodel.network = network
            savemodel.parameters.combinations = combines
            savemodel.genes = names
            # and the model it self 
            savemodel.ODEmodel = make_new_ODEmodel(network,combines,names)
            
            return(savemodel)
        
        else:
            raise TypeError("Input to genesnake.ODEmodel.make_model must be of type numpy.ndarray or pandas.DataFrame.")
    
    
    def _start_ODEmodel(self):
        """
        function for converting the ODEmodel.model from a string to a runing function 
        Takes no input and as output adds the model to the runing_model variable. 
        """
        # make an empty dict to store the dunction in 
        mod_func= {}
    
        # start the function in the dict 
        exec(self.ODEmodel,mod_func)
        # add the function to the class name space 
        setattr(self,"_runing_ODEmodel",mod_func["model"])
        
    
    def train(self):
        """
        function to find steady state from which experiments can be simulated.
        Takes no input as it will lift all parameters from the class
        gives no output other than adding the steady state attributes to the model 
        """
                
        #!# add warnings about missing parameters. 
        
        # if the model is not a function start it up and add it to name space via _start_model
        if self._runing_ODEmodel==None:
            self._start_ODEmodel()
        # set all perturbation to 0 meaning no change
        perts = np.zeros(len(self.genes)).tolist()
        # merge RNA and prot in to a single list so that
        # both can be updated with each ODE time step
        y = self.parameters.start_rna+self.parameters.start_prot
        
        # run the model using odeint with all the set parameters
        ss = odeint(self._runing_ODEmodel,y,self.parameters.time,
                    (self.parameters.alphas,
                     self.parameters.max_rna_exp,
                     self.parameters.max_prot_exp,
                     self.parameters.hill_coef,
                     self.parameters.dis_const,
                     self.parameters.lambda_RNA,
                     self.parameters.lambda_prot,
                     self.parameters.combinations,
                     perts))
        
        # split the RNA and protein back up again for human readability
        # and store these in the GNRmodel
        self.steady_state_RNA = ss[-1,0:len(self.genes)].tolist()
        self.steady_state_prot = ss[-1,len(self.genes):].tolist()
    
    
    def experiment(self):

        # make sure the key elements for the model exists, steady state and perturbation
        if self.steady_state_RNA is None or self.steady_state_prot is None:
            raise ValueError("Attempted to run the experiment function on a modle without a steady state in genesnake.ODEmodel.experiment This will cause unpredictable results as the starting point is random. Please run ODEmodel.train befor attempting to run this function.")
        if self.perturbation is None:
            raise ValueError("Attempted to run the experiment function on a modle without a perturbation matrix in genesnake.ODEmodel.experiment This will not work as there is no experiment describe and thus nothing to simulate. Please run ODEmodel.get_pert befor attempting to run this function.")
        
        # make a 3D numpy array to store that data in
        # this needs to be improved and further sub selection is needed somehow
        data = np.zeros((len(self.genes),self.perturbation.shape[1],len(self.parameters.time)))
        
        # then for each perturbation creata an experiment where the perturbation is
        # applied to the system
        for i in range(self.perturbation.shape[1]):
            
            # merge RNA and prot in to a single list so that both can be updated with each ODE time step
            y = self.steady_state_RNA+self.steady_state_prot
            
            # select the perturbation for this simulation
            perts = self.perturbation[self.perturbation.columns[i]].values.tolist()     
            
            # and run the ODE model from steady state to max time 
            exp = odeint(self._runing_ODEmodel,y,self.parameters.time,
                    (self.parameters.alphas,
                     self.parameters.max_rna_exp,
                     self.parameters.max_prot_exp,
                     self.parameters.hill_coef,
                     self.parameters.dis_const,
                     self.parameters.lambda_RNA,
                     self.parameters.lambda_prot,
                     self.parameters.combinations,
                     perts))
            
            # then add each simulation as a column over all timepoints in the data array  
            for j in range(len(self.parameters.time)):
                data[:,i,j] = exp[j,:len(self.genes)]
        
        # finally transform this to a dict of pandas dataframes with the right annotation
        # to make it somewhat clear what is going on 
        # this needs fixing as it is most likely the slowest part of the ODEmodel script right now
        out_data = dict()
        for i in range(len(self.parameters.time)):
            out_data["time_"+str(i)] = pd.DataFrame(data[:,:,i],index=self.genes,columns=self.perturbation.columns)
        
        return out_data
    
    
    def set_pert(self,scheme,effect=0.5,reps=3,exp=None,noise=None,multipert=2,targets=None,sign=-1):
        """
        Function for generating perturbation design matrixes with n reps of x experiemnts in various schemes. 
        #!# add more detailed documentation somewhere. 
        INPUT: 
            scheme - String which determines design of the perturbations should have. Currently avalible are: 
                     - diag - Each gene is perturbed once per replicate*experiemnts in turn, all perturbations are along the diagonal
                     - multi - Multipert genes are perturbed per replicate*experiments in a way that ensures each gene is perturbed
                               at least once per replicate. Will attempt to add additional perturbation so that each gene is pertrubed
                               in equal number of experiemts where possible.  
                     - combined  - Perturbs genes in a combinatory targeted fashion. Perturbing up to multipert genes that co-regulate
                                  a target in each experiment. 
                     - random - Randomly perturb multipert genes in each experiment. Will attempt to distribute 
                                pertrubations evenly accross all genes. 
                     - all - All genes will be slightly perturbed (between 0.1-0.3) with random sign in a multifactorial design.
                     - targeted - Similar to diagonal but only the genes listed in targets will be perturbed.
                     - scheme - Will print the avalible perturbation designs. 
            effect - the strength of the perturbation as a float in range 0-1, gene expression will be limited to max 1-effect.
                     e.g. max expression at effect 0.7 => 1-0.7=0.3 > expression = max_expression*0.3 or 30% of control max.  
                     Can be given as single float, a tuple with floats for (min_effect,max_effect) or as a list of flaots size 1*exp. 
                     Note for multi or random specific values can only be set per experiment, if a more fine grain value control is 
                     needed it is possible to generate a full matrix and set as perturbation by using ODEmodel.perturbation=pd.DataFrame
            reps - the number of replicates to generate of each experiemts. Note that each replicate will have an identical pert effect
                   and design.
            exp - the number of experiemnts to generate. Similar to reps but effect and pert design change between exp. 
                  NOTE: Not used in diagonal scheme as that needs genes*reps. Default to 1 exp per gene (exp=len(genes in network))
            noise - Float number. The perturbation effect can vary wiht +/-noise. Set to None or 0 for no noise. 
                    E.g. if effect = 0.7 and noise = 0.1 perturbation effect will range from 0.6-0.8 centering on 0.7.
                    Note: This can not change sign of the perturbation (down<=1,up>=1) nor can it cause negative max expression (<0).
            multipert - Int value between 2-#genes. Used in multi and random pert scheme to determine number of perturbations per exp. 
            targets - List of gene names to be perturbed. Used in pert scheme specific to determine which genes will be perturbed. 
                      NOTE to include multiple pertbations in a single experiemnt use a list of lists e.g. [name1,name2,[name2,name1]..]
            sign - +/-1. Used to determine if the perturbation is up or down regulation. 
        
        OUTPUT: 
            ODEmodel.pertrubations - sets the ODEmodel perturbation to the selected design with selected values in a pd.DataFrame. 
        """
        
        # first check that effect is not greater than 1 or less than 0
        if not isinstance(effect,tuple):
            if effect >1.0:
                print("WARNING: Perturbation effect greater than 100% (1.0) in genesnake.ODEmodel.set_pert. Effect have been set to 1.0")
                effect = 1.0
            elif effect < 0.0:
                print("WARNING: Perturbation effect greater than 0% (0.0) in genesnake.ODEmodel.set_pert. Effect have been set to 0.0")
                effect = 0.0
        else:
            if effect[0]>effect[1]:
                raise ValueError("In genesnake.ODEmodel.set_pert wehn supplying effect as a tuple effect[0] must be smaller than effect[1].")
            elif effect[1] >1.0:
                print("WARNING: Perturbation effect greater than 100% (1.0) in genesnake.ODEmodel.set_pert. Effect have been set to 1.0")
                effect[1] = 1.0
            elif effect[0] < 0.0:
                print("WARNING: Perturbation effect greater than 0% (0.0) in genesnake.ODEmodel.set_pert. Effect have been set to 0.0")
                effect[0] = 0.0
                
        # set noise to 0 if it's none simply to make calculations easy
        if noise == None:
            noise=0
                
        # next check so that reps is at least 1
        if reps < 1: 
            print("WARNING: Fewer than 1 replicate (reps<1) detected in genesnake.ODEmodel.set_pert. reps have been set to 1")
            reps = 1
            
        # if the number of experiemnts are not set set it to 1 per gene  
        if exp==None:
            exp=len(self.genes)
            
        # call the pert generator from perturbation_generators and check some needed parameters for them
        if scheme=="diag": 
            setattr(self,"perturbation",pertgen.diagonal(self.genes,effect,reps,noise,sign))
        elif scheme=="multi":
            if multipert < 2: 
                print("WARNING: Cannot perform multiple pertrubations <2  (multipert<2) in genesnake.ODEmodel.set_pert. multipert have been set to 2")
                multipert = 2
            elif multipert > len(self.genes):
                raise ValueError("Cannot include more perturbations than genes in genesnake.ODEmodel.set_pert multi, random or combined perturbation scheme")
                            
            setattr(self,"perturbation",pertgen.multiple_pert(self.genes,effect,reps,exp,noise,sign,multipert))
                
        elif scheme=="random":
            if multipert < 2: 
                print("WARNING: Using multipert<2 in genesnake.ODEmodel.set_pert for random scheme will result in diagonal scehem it is adviced to use multipert>2.")
            elif multipert > len(self.genes):
                raise ValueError("Cannot include more perturbations than genes in genesnake.ODEmodel.set_pert multi, random or combined perturbation scheme")
                            
            setattr(self,"perturbation",pertgen.random_pert(self.genes,effect,reps,exp,noise,sign,multipert))
                
        elif scheme=="all":
            
            setattr(self,"perturbation",pertgen.all_pert(self.genes,effect,reps,exp,noise,sign))
            
        elif scheme=="targeted":
            if targets == None:
                raise ValueError("Failed to find list specififying which genes to perturbe in genesnake.ODEmodel.set_pert. When using scheme specific a list of which genes to perturb must be given in format [name_gene_1...name_gene_n].")

            setattr(self,"perturbation",pertgen.targeted_pert(self.genes,effect,reps,noise,sign,targets))

        elif scheme == "combined":
            if multipert < 2: 
                print("WARNING: Using multipert<2 in genesnake.ODEmodel.set_pert for combined scheme will result in diagonal scehem it is adviced to use multipert>2.")
            elif multipert > len(self.genes):
                raise ValueError("Cannot include more perturbations than genes in genesnake.ODEmodel.set_pert multi, random or combined perturbation scheme")
                            
            setattr(self,"perturbation",pertgen.combined_pert(self.genes,self.network,effect,reps,exp,noise,sign,multipert)) 
            
        elif scheme=="scheme":
            print("""Avalible pertrubation designs are:
            - diag - Each gene is perturbed once per replicate*experiemnts in turn, all perturbations are along the diagonal
            - multi - Multipert genes are perturbed per replicate*experiments in a way that ensures each gene is perturbed
                      at least once per replicate. Will attempt to add additional perturbation so that each gene is pertrubed
                      in equal number of experiemts where possible.
            - combined  - Perturbs genes in a combinatory targetwd fashion. Perturbing up to multipert genes that co-regulate
                          a target in each experiemnt.  
            - random - Randomly perturb genes between 0 and multipert times in each experiment. Will attempt to distribute 
                       pertrubations evenly accross all genes.
            - all - All genes will be slightly perturbed (between 0.1-0.3) with random sign in a multifactorial design.\n
            - targeted - Similar to diagonal but only the genes listed in targets will be perturbed. 
            - scheme - Will print the avalible perturbation designs.""")
        else:
            raise ValueError("""Unknown perturbation design selected in genesnake.ODEmodel.get_pert. Avalible pertrubation designs are:
            - diag - Each gene is perturbed once per replicate*experiemnts in turn, all perturbations are along the diagonal
            - multi - Multipert genes are perturbed per replicate*experiments in a way that ensures each gene is perturbed
                      at least once per replicate. Will attempt to add additional perturbation so that each gene is pertrubed
                      in equal number of experiemts where possible.  
            - combined  - Perturbs genes in a combinatory targetwd fashion. Perturbing up to multipert genes that co-regulate
                          a target in each experiemnt.
            - random - Randomly perturb genes between 0 and multipert times in each experiment. Will attempt to distribute 
                       pertrubations evenly accross all genes.
            - all - All genes will be slightly perturbed (between 0.1-0.3) with random sign in a multifactorial design.\n
            - targeted - Similar to diagonal but only the genes listed in targets will be perturbed. 
            - scheme - Will print the avalible perturbation designs.""")
            
    
    def _json_save(self,file=None):
        """
        Function to save the model to a json file that can later be loaded in to the model. This can both be used 
        to reuse a model and create new models with underlying parameters and such. 
        INPUT: 
            self - a model to be saved. Can not save an empty un-initilized model for obvious reasons
            file - a string with the path to save the json to 
        OUTPUT: 
            saves file.json with all attributes in the ODEmodel. 
        """
        # first create a dict to store all data in after it is sterilized
        save_dict = dict()
        # as the pandas dataframes needs to be turned to numpy each element is loped over and cleande
        for key in self.__dict__.keys():
            # as the _running_model is a function it can not be saved 
            if key == "_runing_ODEmodel":
                continue
            # then as parameters is a bunch of nested attributes we creata a sub dict of that
            elif key == "parameters":
                # start a parameters dict
                pa = dict()
                
                for k in self.__dict__["parameters"].__dict__.keys():
                    # again skip the internals and unsaved objects
                    if k == "_model_self":
                        continue
                    else:
                        pa[k] = self.__dict__["parameters"].__dict__[k]
                # then add that to the save_dict
                save_dict["parameters"] = pa
            # network and perturbation can only be pandas, check if these are indeed pandas.DataFrames
            elif key == "network":
                if isinstance(self.__dict__[key],pd.DataFrame):
                    save_dict["network"] = self.__dict__[key].values.tolist()
                else:
                    save_dict["network"] = self.__dict__[key]
            elif key == "perturbation":
                if isinstance(self.__dict__[key],pd.DataFrame):
                    save_dict["perturbation"] = self.__dict__[key].values.tolist()
                    save_dict["perturbation_header"] = list(self.__dict__[key].columns)
                else:
                    save_dict["network"] = self.__dict__[key]
                    save_dict["perturbation_header"] = None
            # anything else is just a list so it can be saved directly in the save_dict
            else:
                save_dict[key] = self.__dict__[key]
        
        # make the model in to a json object
        save_data = json.dumps(save_dict,indent=4)
        # write to file
        save_file = open(file,"w+")
        save_file.write(save_data)
        save_file.close()
    
    
    def _pickle_save(self,file):
        """
        Function to save the model to a pickle file that can later be loaded in to the model. This can both be used 
        to reuse a model and create new models with underlying parameters and such. 
        INPUT: 
            self - a model to be saved. Can not save an empty un-initilized model for obvious reasons
            file - a string with the path to save the pickle to 
        OUTPUT: 
            saves file.pickle with all attributes in the ODEmodel. 
        """
        # first create a dict to store all data in after it is sterilized
        save_dict = dict()
        # as the pandas dataframes needs to be turned to numpy each element is loped over and cleande
        for key in self.__dict__.keys():
            # as the _running_model is a function it can not be saved 
            if key == "_runing_ODEmodel":
                continue
            # then as parameters is a bunch of nested attributes we creata a sub dict of that
            elif key == "parameters":
                # start a parameters dict
                pa = dict()
                
                for k in self.__dict__["parameters"].__dict__.keys():
                    # again skip the internals and unsaved objects
                    if k == "_model_self":
                        continue
                    else:
                        pa[k] = self.__dict__["parameters"].__dict__[k]
                # then add that to the save_dict
                save_dict["parameters"] = pa
            # network and perturbation can only be pandas, check if these are indeed pandas.DataFrames
            elif key == "network":
                if isinstance(self.__dict__[key],pd.DataFrame):
                    save_dict["network"] = self.__dict__[key].values.tolist()
                else:
                    save_dict["network"] = self.__dict__[key]
            elif key == "perturbation":
                if isinstance(self.__dict__[key],pd.DataFrame):
                    save_dict["perturbation"] = self.__dict__[key].values.tolist()
                    save_dict["perturbation_header"] = list(self.__dict__[key].columns)
                else:
                    save_dict["network"] = self.__dict__[key]
                    save_dict["perturbation_header"] = None
            # anything else is just a list so it can be saved directly in the save_dict
            else:
                save_dict[key] = self.__dict__[key]
        
        file = open(file,"wb+")
        pickle.dump(save_dict,file)
        file.close()
    
    
    def save(self,file=None,kind=None):
        """
        Function that acts as a wraper for the two options to save data from the ODEmodel. 
        It either creates a json file or a pickle file with all parameters. 
        INPUT: 
            file - A file name or a file path to save the data to 
            kind -[optional]- The type of file to create, can be either json or pickle. Default: json
        OUTPUT:
            Function creates a file on the hardrive based on file input 
        """
        
        # make sure the file to save to is defined 
        if file is None:
            raise ValueError("Attempted to save to None in genesnake.ODEmodel.save. Add a file name to save model.")
        
        if kind is None:
            if file.endswith(".json"):
                kind = "json"
            elif file.endswith(".pickle"):
                kind = "pickle"
            else:
                raise ValueError("Unknown file type in genesnake.GNRmodel.save(). Can only save to json or pickle.")
        else:
            raise ValueError("Unknown file type in genesnake.GNRmodel.save(). Can only save to json or pickle.")
        
        if kind=="json":
            self._json_save(file)
        elif kind == "pickle":
            self._pickle_save(file)
        else:
            raise ValueError("Unknown file type in genesnake.GNRmodel.save(). Can only save to json or pickle.")
    
    
    def _pickle_load(file):
        """
        Function for loading a pickle file with the model and parameters in to a ODEmodel object. 
        It is primarily aimed at loading saved ODEmodels from the ODEmodel.save function but will
        work with any pickle file with the right keys present. 
        INPUT: 
            file - the file name or file path of the file to load
        OUTPUT: 
            loadmodel - a ODEmodel object containg the attributes loaded from the pickle file
        """
        
        file = open(file,"rb")
        data = pickle.load(file)
        file.close()
        
        loadmodel = ODEmodel._dict_to_model(data)
        return loadmodel
    
    
    def _json_load(file):
        """
        Function for loading a JSON file with the model and parameters in to a GRNmodel object. 
        It is primarily aimed at loading saved GRNmodels from the GRNmodel.save function but will
        work with any json file with the right keys present. 
        INPUT: 
            file - the file name or file path of the file to load
        OUTPUT: 
            loadmodel - a GRNmodel object containg the attributes loaded from the json file
        """
        file = open(file,"r")
        data = json.load(file)
        file.close()
        
        loadmodel = GRNmodel._dict_to_model(data)
        return loadmodel
    
    
    def _dict_to_model(data):
        """
        Function that turns a loaded dict from a GRNmodel saved model to an actual model 
        INPUT: 
            data - a dictonary with all the attributes to add to a GRNmodel object 
        OUTPUT: 
            loadmodel - a GRNmodel object with all the attributes from the dictonary
        """
        
        # start a model class and return that.
        loadmodel = GRNmodel.__new__(GRNmodel)
        # start a super class of the class 
        super(GRNmodel,loadmodel).__init__()
        # and then start the class it self
        # a bit odd but requiered to make python behave. 
        loadmodel.__init__(loadmodel)
        
        # loop over each key in the loaded data dict  
        for key in data.keys():
            # and add it to the right key in the GRNmodel
            # the parameters attribute has a special loading as it contains multiple subheadings
            if key == "parameters":
                for k in data[key].keys():
                    setattr(loadmodel.parameters,k,data[key][k])
            else:
                setattr(loadmodel,key,data[key])
        
        # once all attributes are set we convert the network and perturbation to pandas
        loadmodel.network = pd.DataFrame(loadmodel.network)
        # if we have genes use these to set the network names
        if len(loadmodel.genes) == loadmodel.network.shape[0]:
            loadmodel.network.index = loadmodel.genes
            loadmodel.network.columns = loadmodel.genes
            
        loadmodel.perturbation = pd.DataFrame(loadmodel.perturbation)
        if len(loadmodel.genes) == loadmodel.perturbation.shape[0]:
            loadmodel.perturbation.index = loadmodel.genes
        if loadmodel.perturbation_header is not None:
            if len(loadmodel.perturbation_header) == loadmodel.perturbation.shape[1]:
                loadmodel.perturbation.columns = loadmodel.perturbation_header
        
        # then remove the perturbation_header attribute as it only exists to save the header
        delattr(loadmodel,"perturbation_header")
        
        # if there is a model and steady state values start the model up 
        if loadmodel.ODEmodel is not None and loadmodel.steady_state_RNA is not None:
            loadmodel._start_ODEmodel()
        
        return loadmodel
    
    
    def load(file=None,kind=None):
        """
        Function that reads a json file, starts a GRNmodel class and parse the data in to the GRNmodel object 
        INPUT: 
            file - the file to be read 
            kind -[optional]- The type of file [json or pickle] to be read. If not given this will be the end of file. 
                              If the end of file is neither .json or .pickle the function will crash. 
        OUTPUT: 
            loadmodel - A GRNmodel object with the data from the json file added to each attribute
        """
            
        if kind is None:
            if file.endswith(".json"):
                kind = "json"
            elif file.endswith(".pickle"):
                kind = "pickle"
            else:
                raise ValueError("Unknown file type in genesnake.GNRmodel.load(). Can only load json or pickle files.")
        elif kind != "json" or kind != "pickle":
            raise ValueError("Unknown file type in genesnake.GNRmodel.load(). Can only load json or pickle files.")
        
        if kind == "json":
            loadmodel = GRNmodel._json_load(file)
        elif kind == "pickle":
            loadmodel = GRNmodel._pickle_load(file)
        else:
            raise ValueError("Unknown file type in genesnake.GNRmodel.load(). Can only load json or pickle files.")
        
        return loadmodel
    
    
    class param(object):
        
        
        def __init__(self,model):
            # start up all the parameters needed
            self.start_rna = list()
            self.start_prot = list()
            self.time = list()
            self.max_rna_exp = list()
            self.max_prot_exp = list()
            self.hill_coef = list()
            self.combinations = dict()
            self.alphas = dict()
            self.dis_const = list()
            self.lambda_RNA = list()
            self.lambda_prot = list()
            self._model_self = model
        
        
        def add_parameters(self,param_dict,fill=False):
            """
            add documentation here
            """
            
            # make a list of keys we expect to see
            approved = list(vars(self).keys())
            approved.remove("_model_self")
            
            # if we are adding the missing parameters make a list of added attributes
            if fill == True:
                seen = list()
            
            # loop over all elements of kwargs and see if they fit any of the expected names
            for key in param_dict.keys():
                # check that there's no unknown in the input 
                if key not in approved:
                    raise ValueError("Attempted to assign unknown parameter "+str(key)+" in genesnake.GRNmodel.parameters.add_parameters. Only ["+", ".join(approved)+"] can be used to assign values to model parameters")
                
                # check that the two weird once are dicts as everything else is a list or numpy array
                if key == "combinations" or key == "alphas":
                    if not isinstance(param_dict[key],dict):
                        raise TypeError("Input to genesnake.GRNmodel.parameters."+key+" must be of type dict.") 
                    # assign each value from the dict to the corresponding attribute
                    setattr(self,key,param_dict[key])
                    # add the key to the list of added attributes
                    if fill == True:
                        seen.append(key)    
                else:
                    if not isinstance(param_dict[key],list):
                        raise TypeError("Input to genesnake.GRNmodel.parameters."+key+" must be of type list.") 
                    # assign each value from the dict to the corresponding attribute
                    setattr(self,key,param_dict[key])
                    # add the key to the list of added attributes
                    if fill == True:
                        seen = list()
            
            # if filling all non added parameters get the missing once and send them to
            # the generate_parameter wraper to be further feed to the right function 
            if fill == True:
                add = list(set(seen).symmetric_difference(approved))
                
                for attr in add:
                    self.generateSP(attr)
        
        
        def generateMP(self):
            """
            Wraper for lazely adding all parameters with default values
            """
            
            # make a list of keys we expect to see
            approved = list(vars(self).keys())
            approved.remove("_model_self")
            
            for attr in approved:
                if len(getattr(self,attr))==0:
                    self.generateSP(attr)
        
        
        def generateSP(self,parameter,**kwargs):
            """
            Wraper function to generate and assign parameters from a number of different functions 
            based on an input string 
            INPUT: 
                parameter - a string with the name of the parameter to be generated 
                **kwargs -[optional]- number of values for the generator to use in format arrg=value,arrg1=value1 
            OUTPUT: 
                None/assigns parameter values to the right class attribute
            """
            
            # make a list of keys we expect to see
            approved = list(vars(self).keys())
            approved.remove("_model_self")
            
            if parameter not in approved:
                    raise ValueError("Attempted to assign unknown parameter "+str(parameter)+" in genesnake.GRNmodel.parameters.generate_parameter. Only ["+", ".join(approved)+"] can be used.")
            
            if len(kwargs)>0:
                setattr(self,parameter,getattr(pargen,'make_%s' % parameter)(**kwargs))
            else:
                if parameter == "combinations":
                    setattr(self,parameter,getattr(pargen,'make_%s' % parameter)(self._model_self.network))
                elif parameter=="alphas":
                    setattr(self,parameter,getattr(pargen,'make_%s' % parameter)(self._model_self.network,self.combinations,self._model_self.genes))
                else:
                    setattr(self,parameter,getattr(pargen,'make_%s' % parameter)(self._model_self.genes))
        
