"""
Script used to generate a heatmap over distrubution in a numpy 2D matrix. 
takes a numpy 2D array and bins as input and returnes a base64 html discription of the figure
"""

import matplotlib
import matplotlib.pylab as plt
import numpy as np
from sklearn.decomposition import PCA
import base64
from io import BytesIO
import pandas as pd
import random

def random_color():

    color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])

    return color


def plot(array,rep_mat,out,direct):
    
    pca = PCA(n_components=2)
    if direct == "col":
        direct_set = "experiments"
        pcomp = pca.fit_transform(np.transpose(array.values))
        
        max_rep = 0
        rep_order = np.zeros(array.shape[1])
        
        for i in rep_mat.keys():
            rep = 1
            where = rep_mat[i]
            for j in where:
                rep_order[j] = rep
                rep+=1
            if rep > max_rep:
                max_rep = rep
        legend_reps = list()
        for i in range(1,max_rep):
            legend_reps.append("rep:"+str(i))
    else:
        pcomp = pca.fit_transform(array.values)
        rep_order = list(array.index)
        max_rep = int(len(rep_order))
        direct_set = "genes"
        legend_reps = list()
        for i in rep_order:
            if i.isnumeric():
                legend_reps.append("Gene:"+str(i))
            else:
                legend_reps.append(i)
    compDf = pd.DataFrame(data = pcomp, columns = ['principal component 1', 'principal component 2'])
    compDf["rep"] = rep_order
    
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA of '+direct_set, fontsize = 20)
    
    colors = ["#e41a1c","#377eb8","#ff7f00","#984ea3","#ffff33","#a65628","#f781bf","#4daf4a"]
    
    if max_rep > len(colors):
        more_cols = max_rep-len(colors)
        for col in range(more_cols):
            colors.append(random_color())

    for rep, color in zip(compDf["rep"].unique().tolist(),colors):
        indicesToKeep = compDf['rep'] == rep
        ax.scatter(compDf.loc[indicesToKeep, 'principal component 1']
               , compDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
        
    if len(legend_reps) <= 5:
        # if there are 5 or less items in the legend just put it where ever as
        # it's not super likely to cover anything. 
        ax.legend(legend_reps)
    elif len(legend_reps) <= 40:
        # if there are 40 items this still fits in the box
        # but needs to be put just at the edge of the plot to make sure
        # it wont cover anything in the plot 
        ax.legend(legend_reps,bbox_to_anchor=(0.98, 1), loc="upper left")
        
    ax.grid()
    
    if out:
        byte_file = BytesIO()
        fig.savefig(byte_file,format="png")
    
        encoder = base64.b64encode(byte_file.getvalue())
        byte_file.close()
        plt.close()
        return(encoder)
    else:
        plt.show()
