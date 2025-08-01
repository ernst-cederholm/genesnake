#!/usr/bin/env python3

"""
Script for calculating a low dimension projection using UMAP and plotting the results based on groupings 
either by columns if a single file is given or each file if a directory is provided. This can also be specified 
by the user but each csv must contain the same column names if grouping by columns.

Dependencies:
 umap-learn  (Note: NOT "umap"!)
 numba sklearn numpy and scipy pandas plotly

By:
 Thomas Hillerton, Erik Sonnhammer, 2023
"""

import plotly.express as px
from umap import UMAP
import pandas as pd
import argparse
import glob
import os


def get_dir_data(directory,sep=",",axis="row"):
    """
    Function for merging all .csv files in directory in to a 
    pandas dataframe adding an extra last column that annotates 
    the file the data comes from originally. 
    Input: 
      - directory - a directory with csv files to read 
      - sep - the seperator to read the csv with. Default \",\"
    Output: 
      - dataframe - a pandas dataframe with all csv files in them 
    """
    
    # get all files in directory
    files = sorted(glob.glob(directory+"/*.csv"), reverse=True)
    # start empty name variable to check if dataframe should be appended
    # or created 
    dataframe = "None"
    
    for file in files:
        df = pd.read_csv(file,sep=sep)
        
        if axis!="row":
            df = df.T
        
        df["file"] = file.split("/")[-1].split(".")[0]
        
        # if the first column are not numbers assume this is the original index
        # as this can be repeated in the files we save it but do not use it as
        # the index here 
        if axis=="row":
            if df.iloc[:,0].dtype == object:
                # change the name to index so we know what it is 
                df.columns = ["ind"]+list(df.columns[1:])
                # and move it to the last position to make it easy to sub select
                # the numerical data 
                df = df[list(df.columns[1:])+["ind"]]
        else:
            # if there were some form of index add this as the new column names
            # if not colum names will simply be 0..n 
            if df.iloc[0,:].dtype == object:
                # move the column names out of the datafram
                df.columns = df.iloc[0,:]
                # check if the first column is still an index 
                if df.iloc[:,0].dtype == object:
                # but this time if it is set it to the index to remove it
                    df = df.set_index(df.columns[0])
                # then create a column to groupby from the columns that are left
                df["ind"] = list(df.columns) 
                
                
        # then append the df to dataframe that will be returned
        # if dataframe is not defined we copy the current df in to it 
        if type(dataframe) == str:
            dataframe = df
        # else we merge them 
        else:
            # first check that the same columns are in both dataframes
            if not set(list(df.columns)) == set(list(dataframe.columns)):
                print("WARNING: found different columns in csvs that are being compared.\nThis might cause issues with the UMAP projection")
            # then let the user deal with that while we concatinate the dataframes
            dataframe = pd.concat([dataframe,df],join="outer",ignore_index=True)
            dataframe = dataframe.reindex()

    # return dataframe
    return(dataframe)


def get_file_data(file,sep=",",axis="row"):
    """
    Function for reading .csv files in to a pandas dataframe 
    adding an extra last column that annotates the file name. 
    Input: 
      - file - a csv files to read
      - sep - the seperater to read the csv with. Default \",\"
    Output: 
      - df - a pandas dataframe with all csv files in them 
    """
    df = pd.read_csv(file,sep=sep)

    if axis!="row":
        df = df.T 
    
    df["file"] = file.split("/")[-1].split(".")[0]
    # if the first column are not numbers assume this is the original index
    # as this can be repeated in the files we save it but do not use it as
    # the index here
    if axis=="row":
        if df.iloc[:,0].dtype == object:
            # change the name to index so we know what it is 
            df.columns = ["ind"]+list(df.columns[1:])
            # and move it to the last position to make it easy to sub select
            # the numerical data 
            df = df[list(df.columns[1:])+["ind"]]
    else:
        if df.iloc[0,:].dtype == object:
            # move the column names out of the datafram
            df.columns = df.iloc[0,:]
            # check if the first column is still an index 
            if df.iloc[:,0].dtype == object:
                # but this time if it is set it to the index to remove it
                df = df.set_index(df.columns[0])
            # then create a column to groupby from the columns that are left
            df["ind"] = list(df.columns) 
            
                
    return(df)


def _umap_plot(data,nneigbors,metric,min_dist,groupby,dim):
    """
    Function for performing umap reduction and plot results 
    Input: 
     - data - the data to work with in a pandas dataframe
     - nneighbors - number of neighbours for umap to use 
     - groupby - name of column to colour plot by or \"column\" to group data based on column names 
                 the column to group by must be after any numerical columns as only those before it will
                 be select. 
     [- dim - string either 2D or 3D. Determines dimensions in plot. Default \"2D\"]
    Output: 
     - fig - a plotly figure object 
     - umap_proj - the output data from the umap model 
    """
    
    # make sure the selected column is in the dataframe
    if groupby not in data.columns:
        raise ValueError("Could not find selected column: "+groupby+" in data.")    
    
    # if the groupby is not a specific column we still check if file or index is in
    # the columns as the script by default adds this
    if groupby =="file":
        until = list(data.columns).index("file")-1
        # and create a seperate dataframe with only the numbers to do UMAP on 
        features = data.loc[:,:data.columns[until]]
    # if file is not found we check for index 
    elif groupby=="ind":
        until = list(data.columns).index("ind")-1
        # and create a seperate dataframe with only the numbers to do UMAP on 
        features = data.loc[:,:data.columns[until]]
    # if it is not we assume that all data should be used 
    else:
        # get all columns before the file and possibly index columns
        # as those two are not numerical
        until = list(data.columns).index(groupby)-1
        # and create a seperate dataframe with only the numbers to do UMAP on 
        features = data.loc[:,:data.columns[until]]

    # select the number of components to use (2 or 3)
    if dim == "2D":
        comps = 2
    elif dim == "3D":
        comps = 3
    else:
        raise ValueError("Unrecognized dimension argument in umap_plot. Only 2D or 3D plots avalible")
        
    # set up the umap model 
    umap = UMAP(min_dist=min_dist, metric=metric, n_neighbors=nneigbor, n_components=comps, init='random', random_state=0)
    # and fit the data
    umap_proj = umap.fit_transform(features)
    
    # then create an image object in fig either 2 or 3D 
    if dim == "2D":
        fig = px.scatter(
            umap_proj, x=0, y=1,
            color=eval("data."+groupby), labels={'color': groupby}
        )
    elif dim == "3D":
        fig = px.scatter_3d(
            umap_proj, x=0, y=1, z=2,
            color=eval("data."+groupby), labels={'color': groupby}
        )
        # if the dim is neither 2 or 3D return an error as there are no dimensions to plot in
    else:
        raise ValueError("Unrecognized dimension argument in umap_plot. Only 2D or 3D plots avalible")
        
    # make the dots a bit clearer and smaller
    fig.update_traces(marker=dict(size=5,line=dict(width=2,color='DarkSlateGrey')),selector=dict(mode='markers'))
    
    # finally return both the projected data and the figure object
    return(fig,umap_proj)


def umap(data,nneigbor,metric="euclidean",min_dist=0.1,out=False,groupby=None,save=False,dim="2D",axis="row",trim=None):
    """
    Function for evaluating input and calling functions 
    should most likely only be used when script is called from the terminal
    as it assumes input are files 
    Input: 
     - data - the data to be plotted either a single csv file or a directory with csv files
     - nneighbors - number of neighbours for umap to use 
     - output - name of file to save the created html file to 
     [- groupby - option to set grouping to columns or file. Default column for single file and file for directory] 
     [- savedata - option to save the data used to create the plot. Default False]
    Output: 
     - html - plot of the projection in 3d
     [- plot_data - the data used to create the plot.]
    """
    # first define if we are working on a pd.Dataframe, a single file or a directory
    if isinstance(data,type(pd.DataFrame)):
        # if the data is already a dataframe do nothing more with it
        plot_data = data
        if not groupby:
            if axis == "row":
                df["ind"] =  list(df.index)
            else:
                df["ind"] =  list(df.columns)
    elif os.path.isdir(data):
        # if it is a directory call the get_dir_data function
        # to merge all csv files in the directory in to a single
        # dataframe with file name annotation. 
        plot_data = get_dir_data(data,axis=axis)    
    else:
        # if it's not a directory we do basically the same thing but only once.
        plot_data = get_file_data(data,axis=axis)       

    # if trimming the groupby remove that many characters from the end of each
    # entry in the groupby column
    if trim:
        plot_data[groupby] = plot_data[groupby].str[:-trim]
        
    # if not specified use the created ind column for groupby
    if not groupby:
        groupby="ind"
    
    # then send this to the umap and ploting function that does most
    # of the work here
    [fig,umap_proj] = _umap_plot(plot_data,nneigbor,metric,min_dist,groupby,dim)
    
    if out:
        # add html to out if it is not there
        if not out.endswith(".html"):
            fig.write_html(out+".html")
        else:
            fig.write_html(out)
    else:
        fig.show(rendere="png")
            
    if save:
        print(umap_proj)
        
    

if __name__ == "__main__":
    # if script is called then parse terminal input
    parser = argparse.ArgumentParser(description='Script for calculating a low dimension projection using UMAP and plotting the results based on groupings either by columns if a single file is given or each file if a directory is provided. This can also be specified by the user but each csv must contain the same column names if grouping by columns.')
    parser.add_argument('data', help="the data to be plotted either a single csv file or a directory with csv files")
    parser.add_argument('n_neighbor', type=int, help="number of neighbours for umap to use")
    parser.add_argument('output', help="name of file to save the created html file to. Will also be used to save data to if savedata is true.")
    parser.add_argument("-g","--groupby",dest="groupby",default=None,help="option to set grouping to columns or file. Default \"column\" for single file and \"file\" for directory")
    parser.add_argument("-s","--savedata",dest="savenum",default=False,type=bool,help="option to save the data used to create the plot. Default False")
    parser.add_argument("-d","--dimension",dest="dim",default="2D",type=str,help="option to set dimensions in scatter plot. Default 2D")
    parser.add_argument("-a","--axis",dest="axis",default="row",type=str,help="Option to project based on column or row. Default row")
    parser.add_argument("-t","--trim",dest="trim",default=None,type=int,help="Option to trim n characters of the end of the groupby column (useful for replicate or file extension data). Default None")
    parser.add_argument("-e","--metric",dest="metric",default="euclidean",type=str,help="Metric used by UMAP, see https://umap-learn.readthedocs.io/en/latest/parameters.html#metric. Default euclidean")
    parser.add_argument("-m","--min_dist",dest="min_dist",default=0.1,type=float,help="Min distance between points. Default 0.1")

    args = parser.parse_args()
    # and store arguments in variables

    data = args.data
    min_dist = args.min_dist
    metric = args.metric
    nneigbor = args.n_neighbor
    out = args.output
    groupby = args.groupby
    save = args.savenum
    dim = args.dim
    axis = args.axis
    trim = args.trim
    
    # check if groupby is set and if not set it based on data
    if not groupby:
        if os.path.isfile(data):
            groupby = "ind"
        elif os.path.isdir(data):
            groupby = "file"
        else:
            raise Exception("Warning "+data+" does not seem to exist")
   
    # once we have all parameters feed them in to main
    umap(data,nneigbor,min_dist,metric,out,groupby=groupby,save=save,dim=dim,axis=axis,trim=trim)
