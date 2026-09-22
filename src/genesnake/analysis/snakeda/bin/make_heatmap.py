import matplotlib.colors as colors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import base64
from io import BytesIO

# TODO: add lables
def plot(df,out):
    """
    Function for plotting a heatmap of expression data.
    INPUT:
        df - matrix with expression data, genesXexperiments
        out - option for writing input to html file
    OUTPUT:
        encoder - encoded heatmap plot, if json/html output is desired
    """
    # sort genes by name for more orderd plot
    df = df.sort_index(axis=0)
    
    fig = plt.figure(figsize=(10,10))
    ax = sns.heatmap(df, linewidths=0, norm=colors.LogNorm(), xticklabels=False, yticklabels=False) #log based color
    plt.xlabel("Experiments")
    plt.ylabel("Genes")
      
    if out:
        byte_file = BytesIO() #create in memory file to write to
        fig.savefig(byte_file,format="png")
    
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