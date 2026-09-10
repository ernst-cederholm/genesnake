import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO

# TODO: add lables
def plot(array,out):
    """
    Function for plotting a histogram of expression data.
    INPUT:
        array - matrix with expression data, genesXexperiments
        out - option for writing input to html file
    OUTPUT:
        encoder - encoded heatmap plot, if json/html output is desired
    """
    plot_ray = array.flatten() # make 2D array 1D 
    bins = 100 #make % of total numbers as bins

    fig = plt.figure(figsize=(10,10))
    plt.hist(plot_ray, bins=bins, histtype="bar")

    if out:
        byte_file = BytesIO() #create in memory file to write to
        plt.savefig(byte_file, format="png")

        if out.endswith(".png"):
            return(byte_file)
        else:        
            encoder = base64.b64encode(byte_file.getvalue()) # encode figure so it can be handled for json/html
            byte_file.close()
            plt.close()
            return(encoder)
    else:
        plt.show()
        plt.close()