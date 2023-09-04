"""
Script used to generate a heatmap over distrubution in a numpy 2D matrix. 
takes a numpy 2D array and bins as input and returnes a base64 html discription of the figure
"""

import matplotlib
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import base64
from io import BytesIO


def plot(array,out):
    
    byte_file = BytesIO()
    
    fig = plt.figure(figsize=(10,10))
    ax = sns.heatmap(array, linewidths=0)
      
    if out:
        byte_file = BytesIO()
        fig.savefig(byte_file,format="png")
    
        encoder = base64.b64encode(byte_file.getvalue())
        byte_file.close()
        plt.close()
        return(encoder)
    else:
        plt.show()
