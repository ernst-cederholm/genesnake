# class for storing a geneSnoake model as a string object and all the parameters needed to use it
# created: 2022-11-08

from .perturbation_generators import pertgen
from .make_new_model import make_new_model
from .parameter_generators import pargen
from scipy.integrate import odeint
import scipy.stats as stats
import pandas as pd
import numpy as np
import itertools
import types


class GRNmodel(object):
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
        self.model = None    
        self.parameters = self.param(self)
        self.network = None
        self.genes = list()
        self._runing_model = None
        self.perturbation = None
    
    def make_model(network,*args,**kwargs):
        """
        add a bunch of documentation here
        """
        
        if isinstance(network,GRNmodel):
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
            
            self.model= make_new_model(network,combines,names)
        
        elif isinstance(network, pd.DataFrame) or isinstance(network, np.ndarray):
            network = network
            i = 0
            if "genes" in kwargs.keys():
                names = kwargs["genes"]
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
            savemodel = GRNmodel.__new__(GRNmodel)
            # start a super class of the class 
            super(GRNmodel,savemodel).__init__()
            # and then start the class it self
            # a bit odd but requiered to make python behave. 
            savemodel.__init__(savemodel)
            # add all attributes that are known 
            savemodel.network = network
            savemodel.parameters.combinations = combines
            savemodel.genes = names
            # and the model it self 
            savemodel.model = make_new_model(network,combines,names)
            
            return(savemodel)
        
        else:
            raise TypeError("Input to genesnake.GRNmodel.make_model must be of type numpy.ndarray or pandas.DataFrame.")
        
        
    def _start_model(self):
        """
        function for converting the GRNmodel.model from a string to a runing function 
        Takes no input and as output adds the model to the runing_model variable. 
        """
        # make an empty dict to store the dunction in 
        mod_func= {}
        # start the function in the dict 
        exec(self.model,mod_func)
        # add the function to the class name space 
        setattr(self,"runing_model",mod_func["model"])
    
    
    def train(self):
        """
        function to find steady state from which experiments can be simulated.
        Takes no input as it will lift all parameters from the class
        gives no output other than adding the steady state attributes to the model 
        """

        # if the model is not a function start it up and add it to name space via _start_model
        if self.runing_model==None:
            self._start_model()
        # set all perturbation to 0 meaning no change
        perts = np.zeros(len(self.genes)).tolist()
        # merge RNA and prot in to a single list so that
        # both can be updated with each ODE time step
        y = self.parameters.start_rna+self.parameters.start_prot
        
        
        # run the model using odeint with all the set parameters
        ss = odeint(self.runing_model,y,self.parameters.time,
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
            raise ValueError("Attempted to run the experiment function on a modle without a steady state in genesnake.GRNmodel.experiment This will cause unpredictable results as the starting point is random. Please run GRNmodel.train befor attempting to run this function.")
        if self.perturbation is None:
            raise ValueError("Attempted to run the experiment function on a modle without a perturbation matrix in genesnake.GRNmodel.experiment This will not work as there is no experiment describe and thus nothing to simulate. Please run GRNmodel.get_pert befor attempting to run this function.")
        
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
            exp = odeint(self.runing_model,y,self.parameters.time,
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
        # this needs fixing as it is most likely the slowest part of the GRNmodel script right now
        out_data = dict()
        for i in range(len(self.parameters.time)):
            out_data["time_"+str(i)] = pd.DataFrame(data[:,:,i],index=self.genes,columns=self.perturbation.columns)
        
        return out_data
    
    
    def get_pert(self,scheme,reps=3,sign="down"):
        """
        add documentation
        """
        setattr(self,"perturbation",getattr(pertgen,'{}'.format(scheme))(self.genes,reps,sign))
        
        
        
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
        
