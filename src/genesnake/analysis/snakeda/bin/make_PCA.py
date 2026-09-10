import matplotlib.pylab as plt
import numpy as np
from sklearn.decomposition import PCA
import base64
from io import BytesIO
import pandas as pd

def plot(data_df, rep_dict, out, direct):
    """
    Function for plotting two component PCA plots with labels depending on which dimention is reduced and if there are replicates.
    INPUT:
        data_df - pandas.DataFrame with gene expression, genesXexperiments
        rep_dict - dictionary for looking up which replicates are for which experiment
        out - option for either showing or encoding the final figure
        direct - which dimention is to be analysed
            - experiments
            - genes
    OUTPUT:
        encoder - b64 encoded figure if the output for the snakeda tool should be a file. Figure is otherwise shown.
    """
    # initialize two component PCA
    pca = PCA(n_components=2)
    max_group = 0
    if direct=="experiments":
        # find principal components for observations
        pca_components = pca.fit_transform(np.transpose(data_df))

        # make list for which replicate observations are part of
        group_markers = np.empty(data_df.shape[1])
        for i in rep_dict.keys(): # loop over experiments
            group = 0
            obs_inds = rep_dict[i]
            for j in obs_inds: # loop over replicates for experiment
                group_markers[j] = group
                group += 1
            if group > max_group:
                max_group = group

    elif direct=="genes":
        pca_components = pca.fit_transform(data_df)
        group_markers = np.zeros(data_df.shape[0])
        max_group += 1

    else:
        raise ValueError("Unknown direction in make_PCA. Supported methods are: experiments and genes")

    component_df = pd.DataFrame(data=pca_components, columns=["Principal component 1", "Principal component 2"])
    component_df["Group"] = group_markers

    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA of '+direct, fontsize = 20)

    if direct=="experiments":
        if max_group>20:
            ax.scatter(component_df.loc[:,"Principal component 1"], 
                    component_df.loc[:,"Principal component 2"])
        else:
            for group in range(max_group):
                ax.scatter(component_df.loc[(component_df.Group==group)]["Principal component 1"], 
                        component_df.loc[(component_df.Group==group)]["Principal component 2"],         
                        label="rep:"+str(group+1)
                        )
                ax.legend(bbox_to_anchor=(0.98, 1), loc="upper left")
    else:
        ax.scatter(component_df.loc[:,"Principal component 1"], 
                component_df.loc[:,"Principal component 2"])

    ax.grid()

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
