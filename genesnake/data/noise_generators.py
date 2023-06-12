"""
A collection of functions that are used to generate noise for the GRNmodle.py script.
Note that they are not imported in to the data namespace only the model class. 
"""

import pandas as pd
import numpy as np
import math

class noise_gen:

    def make_gaussian(data,SNR):
        """
        Function that wraps the numpy.random.normal function for easy use with the geneSNAKE GNRmodel data generation 
        for adding white noise to experiemnts. 
        INPUT: 
            data - numpy.array, the data to add noise to. This is used both to get magnitude of noise and size of noise matrix  
            SNR - int or float, the requested SNR. This value will be used to determine the standard deviation needed to be added 
                  to get approximately that SNR in the output data. 
        OUTPUT: 
            noise - a numpy array of size data with the data+noise. 
        """
        
        # then we use the SNR equation to find the stdE based on the relation SNR = (mean(data)**2)/variance(noise)
        # variance(noise) = stdE(noise)**2
        stdE = np.sqrt((np.mean(data)**2)/(SNR*np.var(data)))
        noise = np.random.normal(0,1,data.shape)       
        
        SNR_scale = np.sqrt(np.var(data)/(SNR*np.var(noise)))
        
        data = data+noise*SNR_scale
        mask = data<=0
        mask = mask*np.random.uniform(0,min(1e-5,np.min(abs(data))),data.shape)
        data[np.where(data<=0)] = 0
        data = data+mask
        return data
    
    
    def make_micro_array_like(data,SNR):
        """
        Model for creating micro array like noise based on the implementation of a similar nature in the GeneNetWeaver paper and the 
        Statistical analysis of MPSS measurements: Application to the study of LPS-activated macrophage gene expression paper by 
        Stolovitzky et al 2005. 
        Note that the output SNR will not be in line with the requested SNR but they will depend on one another in a way such that 
        a lower SNR will lead to higher noise. 
        INPUT:
            data - numpy.array, the data to add noise to. This is used both to get magnitude of noise and size of noise matrix 
            SNR - int or float, the requested SNR. This value will be used to determine the standard deviation needed to be added 
                  to get approximately that SNR in the output data. 
        OUTPUT: 
            noise - a numpy array of size data with the data+noise. 
        """
        
        # create a vectorized function that will be applied to each cell of the numpy array seperatly
        # the noise model is built around finding a variance value dependent on x and multiplying x with this.
        # meaning that small x values will have low noise and large x values have larger noise. 
        f = lambda x: x*np.exp(np.random.normal(0,np.sqrt((x**2)/SNR)))
        # then feed the data in to this function 
        noise = f(data)
        
        return noise

