
from scipy.integrate import solve_ivp
import pandas as pd 
import numpy as np
import math

class simulators(object):
    """
    class that holds various model simulators for easy calling in the GRNmodel class.  
    """
    
    def converged(y):
        """
        Function for testing steady state convergion in ODE model based on multiroot test 
        Based on the GNW function converged with modifications to test across 3 time points instead of 2.
        To ensure stability of the steady-state
        """
        
        # if there are not enough to test reurn false
        if y.shape[1]<=3:
            return False
        
        # values for how much difference is allowed
        # these are stolen right from GNW 
        abs_precision = 0.00001
        rel_precision = 0.001
        
        # check only the last states of the model to save time
        # the assumption here being that once the model reaches SS
        # it cannot break out at any later timepoint
        dxy1 = [abs(y[i,-3]-y[i,-2]) for i in range(y.shape[0])]
        dxy2 = [abs(y[i,-2]-y[i,-1]) for i in range(y.shape[0])]

        # the check if any of the values in these are not steady
        for i in range(y.shape[0]):
            # if the value is not steady accros all 3 time steps (from -3>-2 and -2>-1) return false
            # as it is a time directed process step -3>-1 does not need to be checked.
            
            # add conditions as test# to allow for furter expansion should it be needed  
            T1 = dxy1[i] > abs_precision+rel_precision*abs(y[i,-2])
            T2 = dxy2[i] > abs_precision+rel_precision*abs(y[i,-1])
            
            if any([T1,T2]):
                return False
            
        # if model has not return false all values are assumed to be steady so true is returned 
        return True
    
    
    def find_nth_root(x):
        """
        Function that iteratively tries to find the value n such that 10^n=x. 
        The function is designed to get around the massive rounding error that occurs 
        when using the mathmatical root trick to find n such that n=x^1/10. This value is often 
        off by -1 or more. 
        """
        # start of with n = 1 as the smallest option
        # if only 10 time steps where done time series data will most likely be nonsensical anyway
        high_n = 1
        # define 10 as the base as this is what is used in np.logspace
        base = 10
        # then while 10^n is to small double n 
        while  base** high_n <= x:
            high_n *= 2
        # based on this find the lowest possible end of n 
        low_n = high_n/2
        # and then perform a binary search until 10^n=x
        while True:
            n = (low_n + high_n) / 2
            if low_n < n and 10**n < x:
                low_n = n
            elif high_n > n and 10**n > x:
                    high_n = n
            else:
                return n
    
    
    def _start_protein(model,start_rna):
        #!# should add custom tmax here
        tmin = 0
        tmax = 100
        
        # set solver and error tol, this is just a sanity option to ensure that these are all consistant over all 3
        # possible solver calls
        solver = "LSODA"
        rtol = 1e-8
        
        # make a "perturbation" list of 1s to ensure the max expression is at 100% 
        perts = np.ones(len(model.genes)).tolist()
            
        model.parameters.start_rna = start_rna
        Y = np.array(model.parameters.start_rna+model.parameters.start_prot)
        Y = np.reshape(Y,(-1,1))
        stop = False
        ss_tmin = 0
        ss_tmax = 1
        while not stop and ss_tmax <=100:
            time_steps = np.linspace(ss_tmin,ss_tmax,100).tolist()
            y = Y[:,-1]
            ss = solve_ivp(model._runing_model,(ss_tmin,ss_tmax),y,
                           args=(model.parameters.alphas,
                                 model.parameters.max_rna_exp,
                                 model.parameters.max_prot_exp,
                                 model.parameters.hill_coef,
                                 model.parameters.dis_const,
                                 model.parameters.lambda_RNA,
                                 model.parameters.lambda_prot,
                                 model.parameters.combinations,
                                 perts),method=solver,rtol=rtol,t_eval=time_steps)
            ss_tmin = ss_tmax
            ss_tmax += 1
            Y = np.concatenate((Y,np.reshape(np.array(Y[0:len(model.genes),0].tolist()+ss.y[len(model.genes):,-1].tolist()),(-1,1))),axis=1)
            stop = simulators.converged(Y[len(model.genes):,:]) #some function that depends on ss 
                
        # if we time out and still have not found a stead-state this experiment cannot be used as the rest of the ODE model
        # will be weird. 
        if ss_tmax == 100 and not stop:
            raise GeneratorExit("Failed to find steady state for initial conditions (controls) for ODE model in geneSNAKE.GRNmodel.experiment")
        # split the RNA and protein back up again for human readability
        # and store these in the GNRmodel
        model.steady_state_RNA = Y[0:len(model.genes),-1].tolist()
        model.steady_state_prot = Y[len(model.genes):,-1].tolist()
    
    
    def _run_ODE_sim(model,restart_control,time_points=10,exp_type="ss",draw_pattern="log",end_at_ss=False):
        """
        Internal genesnake function that executes the actual simulation for ODE models. 
        Should not be called directly but rather thought the simulate_data function. 
        Primarily relies on the GRNmodel class to call various parameters. 
        """
        
        #!# should add custom tmax here
        tmin = 0
        tmax = 100
        
        # set solver and error tol, this is just a sanity option to ensure that these are all consistant over all 3
        # possible solver calls
        solver = "LSODA"
        rtol = 1e-8
        
        # if steady_state_RNA is not set create a starting point. 
        if isinstance(model.steady_state_RNA,type(None)) or restart_control == True:
            # make a "perturbation" list of 1s to ensure the max expression is at 100% 
            perts = np.zeros(len(model .genes)).tolist()
            # make a y with RNA and protein start points 
            Y = np.array(model.parameters.start_rna+model.parameters.start_prot)
            Y = np.reshape(Y,(-1,1))
            stop = False
            ss_tmin = 0
            ss_tmax = 1
            while not stop and ss_tmax <= 102:
                time_steps = np.linspace(ss_tmin,ss_tmax,100).tolist()
                y = Y[:,-1]
                ss = solve_ivp(model._runing_model,(ss_tmin,ss_tmax),y,
                               args=(model.parameters.alphas,
                                     model.parameters.max_rna_exp,
                                     model.parameters.max_prot_exp,
                                     model.parameters.hill_coef,
                                     model.parameters.dis_const,
                                     model.parameters.lambda_RNA,
                                     model.parameters.lambda_prot,
                                     model.parameters.combinations,
                                     perts),method=solver,rtol=rtol,t_eval=time_steps)
                ss_tmin = ss_tmax
                ss_tmax += 1
                Y = np.concatenate((Y,ss.y[:,1:]),axis=1)
                stop = False #simulators.converged(Y)
                
            # if we time out and still have not found a stead-state this experiment cannot be used as the rest of the ODE model
            # will be weird. 
            #if ss_tmax == 100 and not stop:
            #    raise GeneratorExit("Failed to find steady state for initial conditions (controls) for ODE model in geneSNAKE.GRNmodel.experiment")

            # split the RNA and protein back up again for human readability
            # and store these in the GNRmodel
            model.steady_state_RNA = Y[0:len(model.genes),-1].tolist()
            model.steady_state_prot = Y[len(model.genes):,-1].tolist()
        
        if model.perturbation is None:
            raise ValueError("Attempted to run the experiment function on a modle without a perturbation matrix in genesnake.GRNmodel.experiment This will not work as there is no experiment describe and thus nothing to simulate. Please run GRNmodel.get_pert befor attempting to run this function.")
        
        # make a variable to add the data to 
        data = None
        
        # then for each perturbation create an experiment where the perturbation is
        # applied to the system
        for i in range(model.perturbation.shape[1]):
            
            # merge RNA and prot in to a single list so that both can be updated with each ODE time step
            Y = np.array(model.steady_state_RNA+model.steady_state_prot)
            Y = np.reshape(Y,(-1,1))            
            # select the perturbation for this simulation
            perts = model.perturbation[model.perturbation.columns[i]].values     
            
            # transform the numpy.array to a list
            perts = perts.tolist()
            
            stop = False
            ss_tmin = 0
            ss_tmax = 1
            
            while not stop and ss_tmax < 102:
                time_steps = np.linspace(ss_tmin,ss_tmax,100).tolist()
                y = Y[:,-1]
                exp = solve_ivp(model._runing_model,(ss_tmin,ss_tmax),y,
                                args=(model.parameters.alphas,
                                      model.parameters.max_rna_exp,
                                      model.parameters.max_prot_exp,
                                      model.parameters.hill_coef,
                                      model.parameters.dis_const,
                                      model.parameters.lambda_RNA,
                                      model.parameters.lambda_prot,
                                      model.parameters.combinations,
                                      perts),method=solver,rtol=rtol,t_eval=time_steps)
                ss_tmin = ss_tmax
                ss_tmax += 1
                
                Y = np.concatenate((Y,exp.y[:,1:]),axis=1)
                if exp_type=="ss" or exp_type=="steady-state" or end_at_ss==True:
                    stop = simulators.converged(Y)
                else:
                    stop = False
            
            # if we time out and still have not found a stead-state print a warning for this experiment and then continue
            if ss_tmax > 100 and not stop and exp_type=="ss":
                print("WARNING: Failed to find steady state for experiment: "+str(i)+" when running ODE model in geneSNAKE.GRNmodel.experiment.")
                print("The last simulated time point will be reported for this experiment.")
                    

            if exp_type=="ss" or exp_type=="steady-state":
                # add the last (and therefore steady-state) time point to the data matrix to be returned
                # adding only the genes not protein information, as that would not be avalible in biological data  
                if isinstance(data,type(None)):
                    data = np.reshape(Y[:len(model.genes),-1],(-1,1))
                else:
                    data = np.concatenate((data,np.reshape(Y[:len(model.genes),-1],(-1,1))),axis=1)
            else:
                # if logspace then:
                if draw_pattern == "log":
                    # note that the logspace starts from 10, this is to allow at least some effect to occur between control and
                    # first experiemnt, and because rint of most floats <10 end up at the same value. 
                    tp = np.rint(np.logspace(1,simulators.find_nth_root(Y.shape[1])-1,num=time_points,base=10))
                    # move the first value in tp to the second place and add in the 1st (right at pert) as the first value
                    tp[1] = tp[0]
                    tp[0] = 1
                else:
                    tp = np.rint(np.linspace(0,Y.shape[1]-1,num=time_points))
                    
                    # remove any duplicate values, this should not be a problem before 100 time points but can cause issues after that
                    tp = np.unique(tp)
                    
                # if time points are lost print a warning
                if tp.shape[0] < time_points:
                    print("WARNING: could only find "+str(tp.shape[0])+" time points out of the requested "+str(time_points)+" time points in geneSNAKE.GRNmodel.experiment. You can either try increasing the total number of time steps for the model by changing the GRNmodel.parameters.time variable, use the reduced number of time points or use a linspace based selection as this is more robust at extreme numbers of time points.") 
                    print("Note that all experiments still has the same time points and should be of equal size/shape!")
                
                # convert tp to a list of ints to use as index 
                tp = tp.astype(int).tolist()
                # add all time points to the data matrix
                if isinstance(data,type(None)):
                    data = Y[:len(model.genes),tp]
                else:
                    data = np.concatenate((data,Y[:len(model.genes),tp]),axis=1)
            
        # finally create names for all experiemnts based on the perturbation matrix
        # and if relevant timepoint
        if exp_type=="ss" or exp_type=="steady-state":
            # for steady-state simply use the perturbation matrix columns 
            exp_names = model.perturbation.columns
        else:
            exp_names = []
            # for time series it is not as trivial as we need to add a t_# for each column
            for i in model.perturbation.columns:
                for j in range(len(tp)):
                    exp_names.append(str(i)+"_T"+str(j+1))
        
        # finally transform this to a pandas dataframes with the right annotation
        out_data = pd.DataFrame(data,index=model.genes,columns=exp_names)
        
        return out_data
