# class for storing a geneSnoake model as a string object and all the parameters needed to use it
# created: 2022-11-08

# import geneSNAKE specific functions 
from .make_new_ODEmodel import make_new_ODEmodel
from .perturbation_generators import pertgen
from .parameter_generators import pargen
from .noise_generators import noise_gen
from .simulators import simulators

# import generic packages
# note that these are seperated to make future maintenence easier as well as making requierments.txt files
import scipy.stats as stats
import pandas as pd
import numpy as np
import itertools
import pickle
import types
import json
import sys


class GRNmodel(object):
    """
    geneSnake class for storing an ODE model as a string object along with all the nessesary parameters to run the model.
    The model class further contains methods to run a model and to generate parameters in and model if needed. 
    """
    
    
    def __init__(self,model=None,parameters=None,network=None,genes=None):
        """
        Function for storing and initilization of a model with corresponding parameters.
        INPUT: 
            model - the source of the model for this object. Input can be: 
                    - None - an empty model class will be created
                    - file.txt - a txt file containg a ODE modle. The file will be read and stored in model.model
                    - str_object - a ODE modle in the form of a string object. It will be stored in model.model. 
        """
        self.steady_state_RNA = None
        self.steady_state_prot = None
        self.model_type = None
        self.model = None    
        self.parameters = self.param(self)
        self.network = None
        self.genes = list()
        self._runing_model = None
        self.perturbation = None
        self.noise_free_data = None
        self.data = None
        self.noise = None
    
    
    def make_model(network,modtype="ODE",generate_parameters=True,**kwargs):
        """
        Function for automatically starting the GRNmodel from a network. Will create a simulation model 
        and corresponding parameters. 
        INPUT:
            network - a pandas dataframe containg the network which to generate data from 
            modtype - the type of model to generate. Currently supports: 
                      - ODE - An ODE based model that simulates gene expression over time through enzyme kinetics - supports time series and steady state.  
                      - line - A linear approximation model that uses the relationship Y = (A**-1)*P to simulate relative change (Y) in the system (A) after perturbation (P). - supports only steady state with output in relative change format.
            generate_parameters - [optional] - option to turn on or off generation of all or some parameters. Takes True, False or a list of parameters to generate. 
            kwargs - optional input to the parameter generator in the from of parameter:{\"option1\":0.1,\"option2\":3}. E.g. combinations={\"max_comb\":3,\"comb_prob\":0.8}.
        To see all options avalible for parameter generation see GRNmodel.params and parameter_generators.py script 
        OUTPUT: 
            returns a running GRNmodel class with a network, a corresponding model, a running model function and all generated parameters. 
        """
        
        network = network
        i = 0
        if "genes" in kwargs.keys():
            names = kwargs["genes"]
        elif isinstance(network, pd.DataFrame):
            names=list(network.columns)
        else:
            names = ["G"+str(i) for i in range(network.shape[0])]

        # clean the names to ensure no characters that will mess with the pseduo code generation occurs
        names = [i.replace(",","") for i in names]
        names = [i.replace("-","") for i in names]
        names = [i.replace("(","") for i in names]
        names = [i.replace(")","") for i in names]
        
        # start a model class and return that.
        savemodel = GRNmodel.__new__(GRNmodel)
        # start a super class of the class 
        super(GRNmodel,savemodel).__init__()
        # and then start the class it self
        # a bit odd but requiered to make python behave. 
        savemodel.__init__(savemodel)
        # add all attributes that are known 
        savemodel.network = network
        savemodel.genes = names
        
        # then add parameters
        if generate_parameters:
            # if making parameters for an ODE model
            if modtype=="ODE":
                # check if generate_parameters is a list in which case use that as param 
                if isinstance(generate_parameters,list):
                    params = generate_parameters
                    # else make a list of all possible parameters 
                else:
                    # start by getting all the parameters from savemodel.parameters
                    params = list(vars(savemodel.parameters).keys())
                    # remove _model_self as this is not generated in the same way 
                    params.remove("_model_self")
                
                # then loop over each 
                for para in params:
                    # check if any parameter settings is in **kwargs 
                    if para in kwargs.keys():
                        # if so feed that dict forward to the attribut setter 
                        savemodel.parameters.generateSP(para,**kwargs[para])
                    # and if not simply use default parameters  
                    else:
                        savemodel.parameters.generateSP(para)
            # add options for future models here
            elif modtype=="line":
                savemodel.parameters = None
        
        savemodel._start_model(modtype)
        return(savemodel)   

    
    def _start_model(self,modtype):
        """
        function for adding the chosen model type to the GRNmodel object and converting the model 
        from a string to a running function
        """
        # get the model
        # Note that this code is primarily here to allow future addition of new models as
        # currently ODE is the only supported model
        if modtype == "ODE":
            self.model= make_new_ODEmodel(self.network,self.parameters.combinations,self.genes)
            self.model_type = "ODE"
        elif modtype == "line":
            self.model="""def model():
            expression = np.matmul(np.linalg.pinv(network),perturbation)"""
            self.model_type = "linear"
        else:
            raise ValueError("""Unknown model type in genesnake.GRNmodel._start_model.
            Currently supported models are:
            - ODE - a differential equations model that simulates change in gene expression over time""")
        
        # and then add this as a running function under the GRNmodel class.
        # Note that this is done regardless of model used as it will later be used to check if
        # simulations can be ran
        # make an empty dict to store the dunction in 
        mod_func= {}
        
        # start the function in the dict 
        exec(self.model,mod_func)
        # add the function to the class name space 
        setattr(self,"_runing_model",mod_func["model"])
            
        
    def start_protein(self,start_rna):
        """
        Function for staring a GRN model with a specific RNA level. aimed at simulating based on real data with a measured RNA control. 
        INPUT: 
            - start_rna - the measured control RNA to simulate the protein levels from. 
        OUT: 
            - sets the model.start_prot and modle.start_rna variables based on the initial simulation. 
        """
        
        # if the GRNmodel is not already running start it up 
        if self._runing_model==None:
            self._start_model()
        
        simulators._start_protein(self,start_rna)
        
                
    def simulate_data(self,restart_control=False,time_points=10,exp_type="ss",draw_pattern="log",noise_model="gaussian",SNR=5,end_at_ss=False):
        """
        Function that act as a wrapper for the set of possible models that could be used for simulation
        of expression data. 
        """
        # if the GRNmodel is not already running start it up 
        if self._runing_model==None:
            self._start_model()
        
        # select what model/solver should be used:
        # Note that this code is primarily here to allow future addition of new models as
        # currently ODE is the only supported model
        if self.model_type == "ODE":
            results = simulators._run_ODE_sim(self,restart_control,time_points=time_points,exp_type=exp_type,draw_pattern=draw_pattern)
        elif self.model_type == "linear":
            results = pd.DataFrame(np.matmul(np.linalg.pinv(self.network.values),self.perturbation.values),index=self.genes,columns=self.perturbation.columns)
        else:
            raise ValueError("""Unknown model type in genesnake.GRNmodel._start_model.
            Currently supported models are:
            - ODE - a differential equations model that simulates change in gene expression over time
            - linear - a simple linear model where change in expression = np.matmul(np.linalg.pinv(network),perturbation)""")        
        # assign the noise free simulation to the GRNmodel
        self.noise_free_data = results
        
        # add noise based on selected models
        # first generate noise with the selected noise model
        if noise_model == "gaussian" or noise_model == "normal":
            [data,noise] = noise_gen.make_gaussian(self.noise_free_data.values,SNR)
        elif noise_model == "micro_array_like" or noise_model == "array_like" or noise_model == "array":
            data = noise_gen.make_micro_array_like(self.noise_free_data.values,SNR)
            noise = data-self.noise_free_data
        elif noise_model == "rnaseq" or noise_model == "rnaseq_like" or noise_model == "coverage_nb":
            noise = noise_gen.make_sequencing_noise(self.noise_free_data.values, SNR)
            data = noise["ratios"] * self.noise_free_data.values
            data = data.to_numpy()
            noise = noise["ratios"].values
        else:
            if noise_model != None:
                print("WARNING unknown noise model selected in genesnake.GRNmodel.simulate_data. While the simulation is fine no noise have been added to your data.")
            data = self.noise_free_data.values
                
        # set all values less than 0 to 0 as counts cannot be negative
        data[np.where(data<0)] = 0
        
        # and assign the noisy data to the GNRmodels data parameter
        self.data = pd.DataFrame(data,index=self.noise_free_data.index,columns=self.noise_free_data.columns)
        self.noise = pd.DataFrame(noise,index=self.noise_free_data.index,columns=self.noise_free_data.columns)
    
    def set_pert(self,scheme,effect=0.5,reps=3,exp=None,noise=None,multipert=2,targets=None,sign=-1):
        """
        Function for generating perturbation design matrices with n reps of x experiemnts in various schemes.
        #!# add more detailed documentation somewhere. 
        INPUT: 
            scheme - String which determines design of the perturbations should have. Currently, avalible are:
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
                     needed it is possible to generate a full matrix and set as perturbation by using GRNmodel.perturbation=pd.DataFrame
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
            GRNmodel.pertrubations - sets the GRNmodel perturbation to the selected design with selected values in a pd.DataFrame. 
        """
        
        # first check that effect is not greater than 1 or less than 0
        if not isinstance(effect,tuple):
            if effect >1.0:
                print("WARNING: Perturbation effect greater than 100% (1.0) in genesnake.GRNmodel.set_pert. Effect have been set to 1.0")
                effect = 1.0
            elif effect < 0.0:
                print("WARNING: Perturbation effect greater than 0% (0.0) in genesnake.GRNmodel.set_pert. Effect have been set to 0.0")
                effect = 0.0
        else:
            if effect[0]>effect[1]:
                raise ValueError("In genesnake.GRNmodel.set_pert wehn supplying effect as a tuple effect[0] must be smaller than effect[1].")
            elif effect[1] >1.0:
                print("WARNING: Perturbation effect greater than 100% (1.0) in genesnake.GRNmodel.set_pert. Effect have been set to 1.0")
                effect[1] = 1.0
            elif effect[0] < 0.0:
                print("WARNING: Perturbation effect greater than 0% (0.0) in genesnake.GRNmodel.set_pert. Effect have been set to 0.0")
                effect[0] = 0.0
                
        # set noise to 0 if it's none simply to make calculations easy
        if noise == None:
            noise=0
                
        # next check so that reps is at least 1
        if reps < 1: 
            print("WARNING: Fewer than 1 replicate (reps<1) detected in genesnake.GRNmodel.set_pert. reps have been set to 1")
            reps = 1
            
        # if the number of experiemnts are not set set it to 1 per gene  
        if exp==None:
            exp=len(self.genes)
            
        # call the pert generator from perturbation_generators and check some needed parameters for them
        if scheme=="diag": 
            setattr(self,"perturbation",pertgen.diagonal(self.genes,effect,reps,noise,sign))
        elif scheme=="multi":
            if multipert < 2: 
                print("WARNING: Cannot perform multiple pertrubations <2  (multipert<2) in genesnake.GRNmodel.set_pert. multipert have been set to 2")
                multipert = 2
            elif multipert > len(self.genes):
                raise ValueError("Cannot include more perturbations than genes in genesnake.GRNmodel.set_pert multi, random or combined perturbation scheme")
                            
            setattr(self,"perturbation",pertgen.multiple_pert(self.genes,effect,reps,exp,noise,sign,multipert))
                
        elif scheme=="random":
            if multipert < 2: 
                print("WARNING: Using multipert<2 in genesnake.GRNmodel.set_pert for random scheme will result in diagonal scehem it is adviced to use multipert>2.")
            elif multipert > len(self.genes):
                raise ValueError("Cannot include more perturbations than genes in genesnake.GRNmodel.set_pert multi, random or combined perturbation scheme")
                            
            setattr(self,"perturbation",pertgen.random_pert(self.genes,effect,reps,exp,noise,sign,multipert))
                
        elif scheme=="all":
            
            setattr(self,"perturbation",pertgen.all_pert(self.genes,effect,reps,exp,noise,sign))
            
        elif scheme=="targeted":
            if targets == None:
                raise ValueError("Failed to find list specififying which genes to perturbe in genesnake.GRNmodel.set_pert. When using scheme specific a list of which genes to perturb must be given in format [name_gene_1...name_gene_n].")

            setattr(self,"perturbation",pertgen.targeted_pert(self.genes,effect,reps,noise,sign,targets))

        elif scheme == "combined":
            if multipert < 2: 
                print("WARNING: Using multipert<2 in genesnake.GRNmodel.set_pert for combined scheme will result in diagonal scehem it is adviced to use multipert>2.")
            elif multipert > len(self.genes):
                raise ValueError("Cannot include more perturbations than genes in genesnake.GRNmodel.set_pert multi, random or combined perturbation scheme")
                            
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
            raise ValueError("""Unknown perturbation design selected in genesnake.GRNmodel.get_pert. Avalible pertrubation designs are:
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
            saves file.json with all attributes in the model. 
        """
        # first create a dict to store all data in after it is sterilized
        save_dict = dict()
        # as the pandas dataframes needs to be turned to numpy each element is loped over and cleande
        for key in self.__dict__.keys():
            # as the _running_model is a function it can not be saved 
            if key == "_runing_model":
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
            saves file.pickle with all attributes in the model. 
        """
        # first create a dict to store all data in after it is sterilized
        save_dict = dict()
        # as the pandas dataframes needs to be turned to numpy each element is loped over and cleande
        for key in self.__dict__.keys():
            # as the _running_model is a function it can not be saved 
            if key == "_runing_model":
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
        Function that acts as a wraper for the two options to save data from the model. 
        It either creates a json file or a pickle file with all parameters. 
        INPUT: 
            file - A file name or a file path to save the data to 
            kind -[optional]- The type of file to create, can be either json or pickle. Default: json
        OUTPUT:
            Function creates a file on the hardrive based on file input 
        """
        
        # make sure the file to save to is defined 
        if file is None:
            raise ValueError("Attempted to save to None in genesnake.GRNmodel.save. Add a file name to save model.")
        
        if kind is None:
            if file.endswith(".json"):
                kind = "json"
            elif file.endswith(".pickle"):
                kind = "pickle"
            else:
                raise ValueError("Unknown file type in genesnake.GNRmodel.save(). Can only save to json or pickle.")
        elif kind=="json":
            self._json_save(file)
        elif kind == "pickle":
            self._pickle_save(file)
        else:
            raise ValueError("Unknown file type in genesnake.GNRmodel.save(). Can only save to json or pickle.")
    
    
    def _pickle_load(file):
        """
        Function for loading a pickle file with the model and parameters in to a model object. 
        It is primarily aimed at loading saved GRNmodels from the GRNmodel.save function but will
        work with any pickle file with the right keys present. 
        INPUT: 
            file - the file name or file path of the file to load
        OUTPUT: 
            loadmodel - a GRNmodel object containg the attributes loaded from the pickle file
        """
        
        file = open(file,"rb")
        data = pickle.load(file)
        file.close()
        
        loadmodel = GRNmodel._dict_to_model(data)
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
        if loadmodel.model is not None and loadmodel.steady_state_RNA is not None:
            loadmodel._start_model()
        
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
            self.hill_coef = dict()
            self.combinations = dict()
            self.alphas = dict()
            self.dis_const = dict()
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
            Wrapper for lazely adding all parameters with default values
            """
            
            # make a list of keys we expect to see
            approved = list(vars(self).keys())
            approved.remove("_model_self")
            
            for attr in approved:
                if len(getattr(self,attr))==0:
                    self.generateSP(attr)
        
        
        def generateSP(self,parameter,**kwargs):
            """
            Wrapper function to generate and assign parameters from a number of different functions
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
                if parameter == "combinations":
                    # if the parameter is combinations use this as it needs a special case 
                    setattr(self,parameter,getattr(pargen,'make_%s' % parameter)(self._model_self.network,**kwargs))
                elif parameter=="alphas" or parameter=="dis_const":
                    # if it is a parameter that should have a value per regulation use this one 
                    setattr(self,parameter,getattr(pargen,'make_%s' % parameter)(self._model_self.network,self.combinations,self._model_self.genes,**kwargs))
                else:
                    setattr(self,parameter,getattr(pargen,'make_%s' % parameter)(self._model_self.genes,**kwargs))
                
            else:
                if parameter == "combinations":
                    # if the parameter is combinations use this as it needs a special case 
                    setattr(self,parameter,getattr(pargen,'make_%s' % parameter)(self._model_self.network))
                elif parameter=="alphas" or parameter=="dis_const":
                    # if it is a parameter that should have a value per regulation use this one 
                    setattr(self,parameter,getattr(pargen,'make_%s' % parameter)(self._model_self.network,self.combinations,self._model_self.genes))
                else:
                    setattr(self,parameter,getattr(pargen,'make_%s' % parameter)(self._model_self.genes))
        
