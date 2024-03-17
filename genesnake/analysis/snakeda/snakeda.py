# need to add:
# way to show SVD
# way to show correlation (some form of heatmap maybe?)

"""
Main script of the EDA tool. Most of the executing code can be found here while several functions are also imported externaly
"""

# general python libraries
from jinja2 import PackageLoader,FileSystemLoader, Environment # used to generate the html report based on manualy created template files
from pandas.api.types import is_string_dtype
from sklearn.ensemble import IsolationForest
from multiprocessing import Pool
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import scipy.stats # used to get a number of statistics out from the data
import argparse
import difflib
import math
import json
import sys
import os

# import genesnake
from genesnake.data import GRNmodel as GRNmodel

# apply a trick to let the relative path .bin needed for importing the script in to geneSNAKE
# and the direct path bin needed to run the script from terminal both exist at the same time
if __name__ == "__main__":
    # plugins for the tool
    from bin import make_PCA
    from bin import pert_rank
    from bin import make_hist
    from bin import make_heatmap
    from bin import pert_simularity_test
else:
    # plugins for the tool
    from .bin import make_PCA
    from .bin import pert_rank
    from .bin import make_hist
    from .bin import make_heatmap
    from .bin import pert_simularity_test


def snr(y,p,norep):
    
    sd = np.zeros(y.shape[0])
    m = np.zeros(y.shape[0])
    if not norep:
        counter = 0
        # loop over all the groups in the perturbation
        for i in p.keys():
            cols = p[i]
            work_data = y.values[:,cols]

            # and find the variance and mean square 
            sd[counter] = abs(np.var(work_data))
            m[counter] = abs(np.mean(work_data)**2)
            counter+=1
            
        SNR = round((np.median(m)/np.median(sd)),4)
    # if no replicates exist simply take the SNR for the whole data at once
    else:
        SNR = (np.mean(y.values)**2)/np.var(y.values)
        
    return(SNR)


def shrink_pert(pert_mat):
    """
    Function for moving pertrubation scheme from full matrix to a dictonary with corresponding columns as values
    This reduces how long the full perturbation matrix must be kept in memory and makes it easier to use the data in later functions
    INPUT:
        pert_mat - a matrix with 0s for non-perturbation and non-zeros for pertrubation 
    OUTPUT: 
        pert_dict - a dictonary of {gene1:[col1,col2,col3],gene2:[col4,col5]}
    """
    
    pert_dict = dict()
    # check what the format of the perturbation matrix is
    values = np.count_nonzero(pert_mat.values)
    zeros = np.count_nonzero(pert_mat.values==0)
    
    # assume if there are more 0s than values that it's a square matrix
    # if not having that many perturbations in column 0 will crash things anyway 
    for row in pert_mat.iterrows():
        if zeros > values:
            pert_dict[row[0]]=np.where(row[1])[0].tolist()
        else:
            pert_dict[row[0]]=row[1].to_list()
        
    return pert_dict


def make_pert(data):
    """
    simple function to try and create a perturbation matrix based on the column names in the data
    INPUT:
        exp_list - a list with experiment names so that reps are clear e.g. G1.1,G1.2..G1.n 
    OUTPUT:
        pert_dict - a dictonary of {gene1:[col1,col2,col3],gene2:[col4,col5]}
    """
    exp_list = list(data.columns)
    # try to check what is the most likely seperator between name and replicate
    dots = sum(1 for i in exp_list if "." in str(i))
    underscore = sum(1 for i in exp_list if "_" in str(i))
    line = sum(1 for i in exp_list if "-" in str(i))
    space = sum(1 for i in exp_list if " " in str(i))

    # check which one is most common 
    seps = {"dots":dots,"underscore":underscore,"line":line,"space":space}
    most = max(seps, key=seps.get)
    
    # and pick that one from a dict
    common_seps = {"dots":".","underscore":"_","line":"-","space":" "}
    sep = common_seps[most]
    
    pert_dict=dict()
    ind = 0
    
    # for each name in the exp_list seperate it on the sep and either add a new list or append
    for i in exp_list:
        i = str(i)
        sepi = i.split(sep)
        if len(sepi)>2:
            raise("\n\nWARNING:SNAKED failed to automatically create perturbation design matrix. Either no replicate seperation is present in the experiment names or the same symbol is used in the name as the seperator e.g. ASD_B_1 or ASD.B.1. Either provide a premade pertrubation design matrix or check the experiment names.\n\n")

        # if the experiment has not been seen create a new key and list
        if sepi[0] not in pert_dict.keys():
            pert_dict[sepi[0]] = [ind]
        # else append the ind value to the list in the dict 
        else:
            pert_dict[sepi[0]].append(ind)
        # by increasing ind with 1 the column number from data
        ind+=1
    
    # try to match the column names with the rows to know which row is perturbed
    # this is only needed for the pert_rank calculation which will not be performed if
    # more than 60% of the perturbations does not match a row
    pert_mat = dict()
    inds = [str(i) for i in data.index]
    for key in pert_dict.keys():
        # first if an exact match exists keep the key
        if key in inds:
            pert_mat[key] = pert_dict[key]
        #else if no exact match exist find one that matches at least 80%
        else:
            closest = difflib.get_close_matches(key,inds,n=1,cutoff=0.8)
            # if a match is found put that as the new key 
            if len(closest)>=1:
                pert_mat[closest] = pert_dict[key]
            # else simply add back the old one 
            else:
                pert_mat[key] = pert_dict[key]
        
    pert_dict = pert_mat
    # check that the replicates makes sense 
    if len(pert_dict.keys()) < 1 or all(len(i)<2 for i in pert_dict.values()):
        raise("\n\nWARNING:SNAKED failed to automatically create perturbation design matrix. Either provide a premade pertrubation design matrix or check the experiment names\n\n")
    
    return pert_dict


def csv_preper(file,perts,norep,direct):
    """
    Function for reading a csv file in to pandas.DataFrame and prepare it for usage in snaked 
    INPUT: 
        file - a csv file with expression data 
        perts - optional pertrubation matrix in csv or pandas.DataFrame format, either in matrix format or memory efficent format with experiment name followed by each relevant column in the data file. E.g. rep1,1,3,5\nrep2,2,4,6.\nNote that if perts are not used SNAKED will try to estimate a perturbation design from row or column name, these should follow naming practis rep1.1,rep1.2..rep1.n.
        direct - direction the replicates in the matrix is expected to be found in in e.g. row for experimentsXgenes [default] or col for genesXexperiments
    OUTPUT: 
        data - a pd.dataframe in format genesXexperiments with the expression data
        pert - a dictonary in format group:[col1,col2,col2] where col corresponds to the relevant column in data 
    """
        
    # read data in to pandas.
    # not that sep=None means pandas will try to check the most likely delimiter using csv.sniffer not no seperation
    # the csv is read without row and column names and these are then added based on if the first row and/or columns are string
    data = pd.read_csv(file,sep=None,header=None,engine="python")
    # first check if the first row is a header 
    if is_string_dtype(data[data.index[0]]):
        headers = data.iloc[0].values
        # recreate the dataframe, this is done so that the datatype of each column is updated once the hearder row is removed
        data = pd.DataFrame(data.values[1:], columns=headers)
    # then check if the first column are strings and set this to the index 
    if is_string_dtype(data[data.columns[0]]):
        data = data.set_index(data.columns[0])
    
    # if the data is in genes in the columns transpose the matrix
    if direct == "col":
        data = data.T
        
    # convert data to float
    data = data.astype(float)
        
    # check if perturbation exists
    if not norep:
        if perts:
            # if perts exist do the same thing as with the data
            pert = pd.read_csv(perts,sep=None,header=None,engine="python")
            # check what the format of the perturbation matrix is
            # set datatype to string as numbers can be strings but strings not numbers
            values = np.count_nonzero(pert.astype(str).values!="0")
            zeros = np.count_nonzero(pert.astype(str).values=="0")
            if zeros > values:
                # first check if the first row is a header 
                if is_string_dtype(pert[pert.index[0]]):
                    headers = pert.iloc[0].values
                    # recreate the dataframe, this is done so that the datatype of each column is updated once the hearder row is removed
                    pert = pd.DataFrame(pert.values[1:], columns=headers)
                    # then check if the first column are strings and set this to the index 
                if is_string_dtype(pert[pert.columns[0]]):
                    pert = pert.set_index(pert.columns[0])        
            else:
                if all(isinstance(i,str) for i in pert.iloc[0].values):
                    headers = pert.iloc[0].values
                    # recreate the dataframe, this is done so that the datatype of each column is updated once the hearder row is removed
                    pert = pd.DataFrame(pert.values[1:], columns=headers)
                if is_string_dtype(pert[pert.columns[0]]):
                    pert = pert.set_index(pert.columns[0])

            if direct == "col":
                pert = pert.T
            
            pert = pert.astype(float)
            pert = shrink_pert(pert)

        else:
            pert = make_pert(data)
    else:
        # if not using perts just return nothing 
        pert=None

    return data,pert


def json_preper(file,perts,norep,direct):
    """
    Function for reading a csv file in to pandas.DataFrame and prepare it for usage in snaked 
    INPUT: 
        file - a json file with expression data 
        perts - optional pertrubation matrix in csv or pandas.DataFrame format, either in matrix format or memory efficent format with experiment name followed by each relevant column in the data file. E.g. rep1,1,3,5\nrep2,2,4,6.\nNote that if perts are not used SNAKED will try to estimate a perturbation design from row or column name, these should follow naming practis rep1.1,rep1.2..rep1.n.
        direct - direction the replicates in the matrix is expected to be found in in e.g. row for experimentsXgenes [default] or col for genesXexperiments
    OUTPUT: 
        data - a pd.dataframe in format genesXexperiments with the expression data
        pert - a dictonary in format group:[col1,col2,col2] where col corresponds to the relevant column in data 
    """

    with open(file, 'r') as f:
        json_data = json.load(f)
        
    # sanitise json keys to be lower case
    json_data =  {k.lower(): v for k, v in json_data.items()}
        
    data = None
    try:
        data = np.array(json_data["data"])
    except KeyError:
        pass
    try:
        data = np.array(json_data["expression"])
    except KeyError:
        raise("\n\nWARNING:SNAKED could not find any data in json file. Expression data must be under key data or expression.\n\n")
        
    data = pd.DataFrame(data)
    
    if direct == "col":
        data = data.T
    
    if "genes" in json_data.keys():
        data.index = json_data["genes"]
    elif "gene" in json_data.keys():
        data.index = json_data["gene"]
    elif "names" in json_data.keys():
        data.index = json_data["names"]
    elif "name" in json_data.keys():
        data.index = json_data["name"]

    if "experiments" in json_data.keys():
        data.columns=json_data["experiments"]
    elif "experiment" in json_data.keys():
        data.columns=json_data["experiment"]

    # check if perturbation exists
    if not norep:
        if perts:
            # if perts exist do the same thing as with the data
            pert = pd.read_csv(perts,sep=None,header=None,engine="python")
            # check what the format of the perturbation matrix is
            values = np.count_nonzero(pert.astype(str).values!="0")
            zeros = np.count_nonzero(pert.astype(str).values=="0")
            if zeros > values:
                # first check if the first row is a header 
                if is_string_dtype(pert[pert.index[0]]):
                    headers = pert.iloc[0].values
                    # recreate the dataframe, this is done so that the datatype of each column is updated once the hearder row is removed
                    pert = pd.DataFrame(pert.values[1:], columns=headers)
                    # then check if the first column are strings and set this to the index 
                if is_string_dtype(pert[pert.columns[0]]):
                    pert = pert.set_index(pert.columns[0])        
            else:
                if all(isinstance(i,str) for i in pert.iloc[0].values):
                    headers = pert.iloc[0].values
                    # recreate the dataframe, this is done so that the datatype of each column is updated once the hearder row is removed
                    pert = pd.DataFrame(pert.values[1:], columns=headers)
                if is_string_dtype(pert[pert.columns[0]]):
                    pert = pert.set_index(pert.columns[0])
                    
            if direct == "col":
                pert = pert.T
            pert = pert.astype(float)
            pert = shrink_pert(pert)
        elif "perturbation" in json_data.keys() or "perturbations" in json_data.keys():
            pert = pd.DataFrame(json_data["perturbation"])
            pert = pert.astype(float)
            pert = shrink_pert(pert)
            
        else:
            pert = make_pert(data)
    else:
        # if not using perts just return nothing 
        pert=None
        
    return data,pert
    
    
def snakeda(data,perts=None,out=None,direct="row",norep=None,ff=None):
    """
    Function for performing basic data analysis and printing it to a html report or stdout. 
    INPUT: 
        data - pandas.DataFrame or file of type csv or json. input should be a matrix of genesXexperiments or experimentsXgenes (see option direct). NOTE for json file data must be named \"data\" or \"expression\". 
        perts - optional pertrubation matrix in csv or pandas.DataFrame format, either in matrix format or memory efficent format with experiment name followed by each relevant column in the data file. E.g. rep1,1,3,5\nrep2,2,4,6.\nNote that if perts are not used SNAKED will try to estimate a perturbation design from row or column name, these should follow naming practis rep1.1,rep1.2..rep1.n.\nFor .json files if pert is not given a perturbation design can be stored in the json file under the name P or perturbation, if this does not exist a similar automatic estimation as with .csv files will be performed using the experiment key.
    out - name of file to save the created html file to. If set to None (default) analysis results will instead be printed stdout. 
    direct - direction the replicates in the matrix is expected to be found in in e.g. row for experimentsXgenes [default] or col for genesXexperiments
    noreps - option to turn of replicates, will disable most metrics but allows for analysis of data with only a single replicate per experiment.
    ff - option to force the file type to be either csv or json. Does nothing if data is a pandas.DataFrame. 
    OUTPUT:
        analysis - the results of the data analysis either to a html report or printed to the stdout.
    """
    
    tables = list() # create an empty list to store the html objects with stats for each file in file list. This will later be added to the report in a for loop based fashion. 
        
    # first determine what the data is
    if isinstance(data,type(pd.DataFrame())) or isinstance(data,GRNmodel.GRNmodel):
        if isinstance(data,GRNmodel.GRNmodel):
            M = data
            data = M.data
            perts = M.perturbation
            
        # need to get the perts if not availeble
        df = data
        # check if perturbation exists and should do so
        if not norep:
            if isinstance(perts,type(pd.DataFrame())):
                rep_mat = shrink_pert(perts) 
            elif isinstance(data,str):
                # if perts exist do the same thing as with the data
                pert = pd.read_csv(perts,sep=None,header=None,engine="python")
                # check what the format of the perturbation matrix is
                values = np.count_nonzero(pert.astype(str).values!="0")
                zeros = np.count_nonzero(pert.astype(str).values=="0")
                if zeros > values:
                    # first check if the first row is a header 
                    if is_string_dtype(pert[pert.index[0]]):
                        headers = pert.iloc[0].values
                        # recreate the dataframe, this is done so that the datatype of each column is updated once the hearder row is removed
                        pert = pd.DataFrame(pert.values[1:], columns=headers)
                        # then check if the first column are strings and set this to the index 
                    if is_string_dtype(pert[pert.columns[0]]):
                        rep_mat = pert.set_index(pert.columns[0])        
                else:
                    if all(isinstance(i,str) for i in pert.iloc[0].values):
                        headers = pert.iloc[0].values
                        # recreate the dataframe, this is done so that the datatype of each column is updated once the hearder row is removed
                        rep_mat = pd.DataFrame(pert.values[1:], columns=headers)
                    if is_string_dtype(pert[pert.columns[0]]):
                        rep_mat = pert.set_index(pert.columns[0])
                if direct == "col":
                    rep_mat = rep_mat.T
                    rep_mat = rep_mat.astype(float)
                rep_mat = shrink_pert(rep_mat)
            elif "perturbation" in json_data.keys() or "perturbations" in json_data.keys():
                pert = pd.DataFrame(json_data["perturbation"])
                pert = pert.astype(float)
                rep_mat = shrink_pert(pert)
                
            else:
                rep_mat = make_pert(data)
        else:
            # if not using perts just return nothing 
            pert=None
            
    elif isinstance(data,str):
        # if the input is a string (and thus likely a file path) determine the file type
        if data.endswith(".csv") or ff == "csv":
            [df,rep_mat] = csv_preper(data,perts,norep,direct)
        elif data.endswith(".json") or ff == "json":
            [df,rep_mat] = json_preper(data,perts,norep,direct)
        else:
            # if the file is neither csv or json perform controlled crash. 
            raise TypeError("\n\nWARNING:unknown input format in SNAKED analysis tool, make sure input is a pandas.DataFrame, or a file of type .csv or .json\n\n")
    else:
        raise TypeError("\n\nWARNING:unknown input format in SNAKED analysis tool, make sure input is a pandas.DataFrame, or a file of type .csv or .json\n\n")

    # make sure the row and column headers are strings 
    df.index = [str(i) for i in df.index.to_list()]
    df.columns = [str(i) for i in df.columns.to_list()]
    
    # make some plots 
    distrubution = make_hist.plot(df.values,out)
    heatmap = make_heatmap.plot(df.values,out)
    if not norep:
        col_pca = make_PCA.plot(df,rep_mat,out,"col")
        row_pca = make_PCA.plot(df,rep_mat,out,"row")
    else:
        pca = None
            
    # calculate several sets of basic statistics for the numerical values
    stats = {"value":[
        df.shape[0]*df.shape[1], # "elems"
        len(np.unique(df.values)), # "uniq"     
        np.count_nonzero(df.values==0.0), # "zeroes"
        str(df.values.shape[0])+", "+str(df.values.shape[1]),
    ],"%":[
        "",
        round((len(np.unique(df.values))/(df.values.shape[0]*df.values.shape[1]))*100,2), # "elems" "uniq"     
        round((np.count_nonzero(df.values==0)/(df.values.shape[0]*df.values.shape[1]))*100,2), # "elems" "zeroes"
        "", # "elems" "nozero"
    ]} 
    
    stats1 = {"value":[
        round(np.mean(df.values),4), # "mean"
        round(np.median(df.values),4), # "median"
        round(np.amax(df.values),4), # "max"
        round(np.amin(df.values),4), # "min"
    ]}
    stats2 = {"value":[
        round(np.var(df.values),4), # "variance"
        round(np.std(df.values),4), # "std", # "standard deviation"
        round(scipy.stats.skew(df.values).mean(),4), #"skew"
        round(snr(df,rep_mat,norep),4), # "snr_wiki"
    ]}
    
    svd = np.linalg.svd(df.values)[1]
    stats3 = {"value":[
        round(np.linalg.cond(df.values),4),
        round(np.max(svd),4),
        round(np.median(svd),4),
        round(np.min(svd),4),
    ]}
    if not norep: 
        # get the stats for the perturbation rank 
        [median_pert_rank,median_percent_rank_pert,times_smallest,times_greatest] = pert_rank.calc(df,rep_mat)
        
        stats4 = {"value":[
            median_pert_rank,
            "",
            times_smallest,
            times_greatest
        ],"%":[
            "",
            median_percent_rank_pert*100,
            round(times_smallest/df.shape[1],4)*100,
            round(times_greatest/df.shape[1],4)*100
        ]}
        
        [median_repcorr,median_allcorr,greater_repocorr,ttest_sig_greater_repcorr] = pert_simularity_test.test(df,rep_mat)
        
        stats5 = {"value":[
            median_repcorr,
            median_allcorr,
            greater_repocorr,
            ttest_sig_greater_repcorr
        ]}
    else:
        # if not using replicates return stat block 4 and 5 as empty
        # this is done only so the printing functions don't crash 
        stats4 = {"value":[
            "  ",
            "  ",
            "  ",
            "  "
        ]}
        
        stats5 = {"value":[
            "  ",
            "  ",
            "  ",
            "  "
        ]}
                
    if out:
        # initiate all the html stuff 
        script_path = os.path.abspath(__file__) # get the path to where this script is to find templates
        script_path = script_path[0:-10]

        # Configure Jinja and ready the loader
        env = Environment(loader=FileSystemLoader(searchpath=script_path+"templates"))

        # set up the templates used for adding to the file
        base = env.get_template("base_report.html")
        stat_section = env.get_template("stats_and_images.html")

        hist_html = None
        rpca_html = None
        cpca_html = None
        heat_html = None
        hist_html = '<img src="data:image/png;base64, {}">'.format(distrubution.decode('utf-8'))
        rpca_html = '<img src="data:image/png;base64, {}">'.format(row_pca.decode('utf-8'))
        cpca_html = '<img src="data:image/png;base64, {}">'.format(col_pca.decode('utf-8'))
        heat_html = '<img src="data:image/png;base64, {}">'.format(heatmap.decode('utf-8'))
        
        # add stats and plots to html list one for each file in file_list
        stats_df = pd.DataFrame(stats, index=["Elements","Unique values","#zeros","matrix size"])
        stats_html = stats_df.to_html()
        
        stats_df = pd.DataFrame(stats1, index=["mean","median","max","min"])
        stats1_html = stats_df.to_html()
        
        stats_df = pd.DataFrame(stats2, index=["variance","stdev","skewness","SNR"])
        stats2_html = stats_df.to_html()
        
        stats_df = pd.DataFrame(stats3, index=["condition value","max SVD","median SVD","min SVD"])
        stats3_html = stats_df.to_html()
        
        stats_df = pd.DataFrame(stats4, index=["median pertrank","median percentile pertrank","# pertrank lowest","# pertrank highest"])
        stats4_html = stats_df.to_html()
        
        stats_df = pd.DataFrame(stats5, index=["median replicate correlation","median dataset correlation","# repcorr>dataset corr","P-val replicate corr > data corr (ttest)"])
        stats5_html = stats_df.to_html()

        # create a name this is done if data is a pandas.dataframe
        if isinstance(data,type(pd.DataFrame())):
            name = "matrix"
        else:
            name = data
                    
         # add stats and plots to html list one for each file in file_list
        tables.append(stat_section.render(
            file="EDA analysis of "+name,
            t1=stats_html,
            t2=stats1_html,
            t3=stats2_html,
            t4=stats3_html,
            t5=stats4_html,
            t6=stats5_html,
            hist=hist_html,
            rpca=rpca_html,
            cpca=cpca_html,
            heat=heat_html
        ))

        # if the out file does not end with .html append this as the file will not opened as code and not a report without that
        if not out.endswith(".html"):
            out = out+".html"
          
            
        writer = open(out, "w+") # open output file for writing
        writer.write(base.render(
            title=out,
            sections=tables
        ))
        
    else:
        # if not making a html file simply merge all stast tables and print them. 
        stats_df = pd.DataFrame(stats, index=["Elements","Unique values","#zeros","matmatrix size"])
        stats_df = pd.concat([stats_df,pd.DataFrame(stats1, index=["mean","median","max","min"])])
        stats_df = pd.concat([stats_df,pd.DataFrame(stats2, index=["variance","stdev","skewness","SNR"])])
        stats_df = pd.concat([stats_df,pd.DataFrame(stats3, index=["condition value","max SVD","median SVD","min SVD"])])
        stats_df = pd.concat([stats_df,pd.DataFrame(stats4, index=["median pertrank","median percentile pertrank","# pertrank lowest","# pertrank highest"])])
        stats_df = pd.concat([stats_df,pd.DataFrame(stats5, index=["median replicate correlation","median dataset correlation","# repcorr>dataset corr","P-val replicate corr > data corr (ttest)"])])
        print(stats_df)


if __name__ == "__main__":
    # if script is called then parse terminal input
    parser = argparse.ArgumentParser(description='Script for performing a general data analysis of gene expression data with a focus on comparing expression across replicates',formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('data', help="the data to analyse, can either be a csvjson file with a matrix of genesXexperiments or experimentsXgenes (see option --direction for details on this).\nNOTE for json file data must be named \"data\" or \"expression\".") 
    parser.add_argument("-p","--perts",dest="pert",default=None,help="optional pertrubation matrix in csv format, either in matrix format or memory efficent format with experiment name followed by each relevant column in the data file. E.g: \nrep1,1,3,5\nrep2,2,4,6.\nNote that if perts are not used SNAKED will try to estimate a perturbation design from row or column name,\nthese should follow naming practis rep1.1,rep1.2..rep1.n.\nFor .json files if pert is not given a perturbation design can be stored in the json file under the name P or perturbation, \nif this does not exist a similar automatic estimation as with .csv files will be performed using the experiment key")
    parser.add_argument("-o","--out",dest="output",default=None,help="name of file to save the created html file to. If set to None (default) analysis results will instead be printed stdout")
    parser.add_argument("-d","--direction",dest="direction",default="row",type=str,help="direction the replicates in the matrix is expected to be found in in e.g. row for experimentsXgenes [default] or col for genesXexperiments")
    parser.add_argument("-n","--noreps",dest="noreps",default=None,help="option to turn of replicates, will disable most metrics but allows for analysis of data with only a single replicate per experiment")
    parser.add_argument("-f","--forcefile",dest="force_file_type",default=None,type=str,help="option to force the file type to be either csv or json.")

    # move input to args
    args = parser.parse_args()
    # and store arguments in variables
    data = args.data
    perts = args.pert
    out = args.output
    direct = args.direction
    norep = args.noreps
    ff = args.force_file_type
    
    # then run the eda tool based on input
    # Note if runing from within another script the eda function has the same
    # defaults as the argparse above 
    snakeda(data,perts,out,direct,norep,ff)
