import pandas as pd
import numpy as np
import scipy.stats
from genesnake.data.GRNmodel import GRNmodel as GRNmodel
from jinja2 import FileSystemLoader, Environment
import os
import argparse
from math import floor, log10, isnan
import json
from PIL import Image

# import helper functions for plotting and more involved analyses
# check how the script runs for how to import them
if __name__ == "__main__":
    from bin import make_heatmap
    from bin import make_histogram
    from bin import make_PCA
    from bin import rep_corr_test
    from bin import rep_corr_test_lrg
    from bin import pert_rank
else:
    from .bin import make_heatmap
    from .bin import make_histogram
    from .bin import make_PCA
    from .bin import rep_corr_test
    from .bin import rep_corr_test_lrg
    from .bin import pert_rank

def round_significant(number, significant):
    """
    Function for rounding to specified number of significant digits.
    INPUT:
        number - number to be rounded
        signigicant - number of significant digits
    OUTPUT:
        rounded number"""
    if number == 0:
        return 0
    elif number == np.nan or isnan(number):
        return np.nan
    else:
        a = log10(abs(number))
        return round(number, significant-1-int(floor(a)))

def signal_to_noise_ratio(data, rep_dict, norep):
    """
    Function to calculate the signal to noise ratio as mean over standard deviation.
    INPUT:
        data - pandas.DataFrame with data
        rep_dict - dictionary to look up which columns are replicates
        norep - marker for if there are replicates or not
    OUTPUT:
        SNR - calculated signal to noise ratio"""

    if not norep:
        stdev = np.zeros(len(rep_dict))
        average = np.zeros(len(rep_dict))

        # loop over the groups of replicates
        for i, pert_gene in enumerate(rep_dict.keys()):
            cols = rep_dict[pert_gene]
            observations = data.iloc[:, cols]

            # absolute values of standard deviation and mean
            stdev[i] = np.std(observations.values)
            average[i] = abs(np.mean(observations))
        SNR = np.mean(average)/np.mean(stdev)
    # case without replicates which calculate SNR on the whole data at once
    else:
        SNR = abs(np.mean(data.values))/np.std(data.values)
    return SNR

def match_experiment_with_gene_name(gene_names, rep_dict):
    """
    Function which tries to match experiments and gene names to update replicate lookup dictionary with
    INPUT:
        gene_names - list with gene names, from index of expression matrix
        rep_dict - dictionary with experiment names (without marker for replicate) as keys
    OUTPUT:
        new_rep_dict - dictionary with updated keys if matches are found
    """
    print("\nTrying to match gene and experiment names for perturbation rank analysis\n")

    new_rep_dict = {}
    for key in rep_dict.keys():
        # if an exact match can be found, keep it
        if key in gene_names:
            new_rep_dict[key] = rep_dict[key]
        # if an exact match does not exist, try to find a close one
        else:
            #closest = difflib.get_close_matches(key, gene_names, n=1, cutoff=0.8)
            # if a close match is found, use as new key
            #if len(closest)==1:
            #    new_rep_dict[closest[0]] = rep_dict[key]
            # if a match could not be found, use the old key
            #else:
                new_rep_dict[key] = rep_dict[key]
    return new_rep_dict

def make_reps_lookup_dict(data_df, norep, method='single_pert', synthetic=False):
    """
    Function for creating a lookup dictionary of which columns are replicates of the same experiment.
    INPUT:
        data_df - pandas.DataFrame with either expression data or perturbation information, 0 for non-perturbed, non-zero for perturbed
        norep - options for if there are replicates or not
        method - option for method to determine which observations are replicates
            - single_pert - will assume each time a gene is perturbed it is a replicate of the other times it is perturbed. Used when data_df is a perturbation matrix where a single gene is perturbed at a time. Groups of replicates will be named after gene perturbed.
            - single_name - will look for common separators in the observetion lables for sign of replicates. Assumes a single gene is perturbed at a time and tries to match experiment lables with gene names.
            - multi_name - will look for common separators in the observation lables for sign pf replicates.
    OUTPUT:
        rep_dict - dictionary with keys labled as genes or experiments, depening on method, and values as column names of replicates.
        norep - potentially updated information for if there are replicates
    """
    rep_dict = dict()
    norep_check = False # marker to check if replicates exist, used to potentially update norep
    if method == 'single_pert':
        for row in data_df.iterrows():
            rep_dict[row[0]] = np.where(row[1])[0].tolist()
        # check to ensure norep is set to True if no replicates are found in data
        if data_df.shape[0] == data_df.shape[1]:
            norep = True
    elif method == 'single_name' or method == 'multi_name':
        # only look for replicates if they are believed to exist
        if not norep:
            col_lables = list(data_df.columns)

            # check common separators between experiment name and replicate tag 
            common_separators = {"dots":".","underscore":"_","dash":"-","space":" "}
            if synthetic:
                common_separators["capital_r"] = "R"

            num_separators = dict()
            for sep_name, sep in common_separators.items():
                num_separators[sep_name] = sum(1 for i in col_lables if sep in str(i))

            # check for the most common one
            most_common_separator = common_separators[max(num_separators, key=num_separators.get)] # note that if multiple separators occurs equal amount of times and are the most common, the first one in the dictionary will be picked and replicate identification can be unreliable

            # for each column label, separate it, check so there are not more than two parts after sepatation, add to rep_dict
            for i, col in enumerate(col_lables):
                col = str(col)
                separated_label = col.split(most_common_separator)
                if len(separated_label)>2: # single part after separation will occur if replicates are labeled the same and is therefore accepted and handled.
                    raise ValueError("\nWARNING:SNAKEDA failed to identify replicates from lables. Either no common separator is used before replicate identifyer or the same symbol is used in the name as the separator e.g. ASD_B_1 or ASD.B.1. Please update lables or use option for no replicates.") 

                # if the experiment has not been seen before, create new key and list
                if separated_label[0] not in rep_dict.keys():
                    rep_dict[separated_label[0]] = [i]
                # else, append the column index to the list in the dictionary
                else:
                    rep_dict[separated_label[0]].append(i)
                    norep_check = True # update marker to show that replicates were found
        else:
            for i, name in enumerate(data_df.columns.to_list()):
                rep_dict[name] = [i]

        if method == 'single_name':
            gene_names = [str(i) for i in data_df.index.to_list()]
            rep_dict = match_experiment_with_gene_name(gene_names, rep_dict)
        # check to ensure norep is set to True is no replicates are found in data
        if not norep_check:
            norep = True
    else:
        raise ValueError("Unknown method type in snakeda.make_reps_lookup_dict. Supported methods are: single_pert, single_name, and multi_name")
    return rep_dict, norep

def csv_prepper(file, direction="genes"):
    """
    Function for reading csv files, converting to pandas.DataFrames and automatically detecting if column and row labels exists, handles cases for none or both of them existing.
    INPUT:
        file - a csv file
        direction - if labels are to be generated, option to ensure that they are applied to the correct axis
            - genes - if genes are on the rows
            - experiments - if experiments are on the rows
    OUTPUT:
        df - a pandas.DataFrame with column and rown labels, from the file is they exist, otherwise generated"""
    # read data into dataframe
    df = pd.read_csv(file, sep=None, header=None, engine="python") # sep=None set reader to automatically detect the separator used, therefor the engine is set to python. header=None to make it possible to handle data with and without a row with lables

    # check first row for column labels and if they do set column and row names to first row and column
    if pd.api.types.is_string_dtype(df.iloc[0][1:]):
        col_names = df.iloc[0].values
        # update dataframe with column lables and remove row where they are taken from
        df = pd.DataFrame(df.values[1:], columns=col_names)

        # set first column as index
        df = df.set_index(df.columns[0])
    else:
        if direction == "experiments":
            df.columns = ["G"+str(i) for i in range(1, df.shape[1]+1)]
            df.index = ["Exp"+str(i) for i in range(1, df.shape[0]+1)]
        elif direction == "genes":
            df.columns = ["Exp"+str(i) for i in range(1, df.shape[1]+1)]
            df.index = ["G"+str(i) for i in range(1, df.shape[0]+1)]
        else:
            raise ValueError("Unknown direction alternative in snakeda. Supported designs are: genes and experiments")

    # convert data to float
    df = df.astype("float")
    return df

def json_to_html(file, stats_data=None):
    """
    Fuction for taking statistics and encoded figures from json file and output as a html file for easier viewing.
    INPUT:
        file - filename without extention or .html if data should come from stats_data
        stats_data - optional input for when data should not be read from json"""

    # check if data should be loaded from json or use dictionary from input
    if not file.endswith(".html"):
        with open(file+".json", "r") as f:
            stats_data = json.load(f)
        file = file + ".html"

    stats_list = stats_data["Statistics"]
    plot_dict = stats_data["Plots"]

    script_path = os.path.abspath(__file__) # get the path to where this script is to find templates
    script_path = script_path[0:-10]
    
    # Configure Jinja and ready loader
    jinja_env = Environment(loader=FileSystemLoader(script_path+"templates"))
    
    # set up html templates
    base = jinja_env.get_template("base_report.html")
    stat_sections = jinja_env.get_template("stats_and_images.html")

    # format figures
    hist_html = "<img src='data:image/png;base64, "+ plot_dict["Histogram"] +"'>"
    epca_html = "<img src='data:image/png;base64," + plot_dict["Experiment PCA"] + "'>"
    gpca_html = "<img src='data:image/png;base64," + plot_dict["Gene PCA"] + "'>"
    heat_html = "<img src='data:image/png;base64," + plot_dict["Heatmap"] + "'>"
    
    # format statistics
    stats = pd.DataFrame(stats_list, index=["Value"]).transpose()

    # add stats and plots to list rendered according to template
    tables = list()
    tables.append(stat_sections.render(
        file = "EDA analysis: " + file,
        t = stats.to_html(),
        hist=hist_html,
        gpca=gpca_html,
        epca=epca_html,
        heat=heat_html
        ))
    
    # write to file according to base template render
    with open(file, 'w+') as outfile:
        outfile.write(base.render(
        title=file,
        sections=tables
        ))       

def snakeda(data, perts_df=None, norep=False, pert_design='single', out=None, direction="genes", synthetic=False):
    """
    Function for basic data analysis.
    INPUT:
        data - pandas.DataFrame, GRNmodel or csv file. matrix with gene expression, typically on the form genesXexperiments
        perts_df - optional perturbation matrix as pandas.DataFrame or csv file, only used if pert_design is set to single
        norep - option for when there are no replicates. Will limit some analyses, but makes it possible to run the rest
        pert_design - option to change between designs with a single and multiple perturbed genes per experiment/observation
            - single - one gene perturbed per observation
            - multiple - one or more genes perturbed per observation
        out - option to write output to file, extention alternatives:
            - none - will write to both json and html
            - json - will onlt write output to json, with figures encoded
            - html - will only write to html file and bypass json
            - png - will make one png file with all figures and write statistics to json file
        direction - option to specify if genes or experiments are on the rows
            - genes - default
            - experiments
        synthetic - option for if the data has naming standard of synthetic data
    OUTPUT:
        analysis - dictionary with statistics and figures show in standard output. If specified the same information as json and/or html file, in the case of json file the images will be encoded to a string"""

    # Cutoff size so large matrices do not run into memory capacity problems
    cutoff = 7500

    # Determine data type and format uniformly
    if isinstance(data, GRNmodel):
        data_df = data.data
        perts_df = data.perturbation
    elif isinstance(data, pd.DataFrame):
        data_df = data
    elif isinstance(data, str):
        if data.endswith(".csv"):
            data_df = csv_prepper(data, direction)
        else:
            raise TypeError("\nWARNING:unknown input format in SNAKED analysis tool, make sure input is a pandas.DataFrame, genesnake.GRNmodel, or csv file with the .csv extention in the name\n")
    else: 
        raise TypeError("\nWARNING:unknown input format in SNAKED analysis tool, make sure input is a pandas.DataFrame, genesnake.GRNmodel, or csv file\n")

    # Ensure the row and column lables are strings
    data_df.index = [str(i) for i in data_df.index.to_list()]
    data_df.columns = [str(i) for i in data_df.columns.to_list()]

    # Determine if perts_df needs to be read from csv and do if needed
    if perts_df==None:
        pass
    elif isinstance(perts_df, pd.DataFrame):
        if perts_df.shape != data_df.shape:
            perts_df = None
            print("\nWARNING: Data and perturbation matrix of different shapes. Will try to find replicates from experiment names.")
    elif isinstance(perts_df, str):
        if perts_df.endswith(".csv"):
            perts_df = csv_prepper(perts_df, direction)
            if perts_df.shape != data_df.shape:
                        perts_df = None
                        print("\nWARNING: Data and perturbation matrix of different shapes. Will try to find replicates from experiment names.")
        else:
            raise TypeError("\nWARNING:unknown input format in SNAKED analysis tool, make sure input perts_df is a pandas.DataFrame, or csv file\n")
    else: 
        raise TypeError("\nWARNING:unknown input format in SNAKED analysis tool, make sure input perts_df is a pandas.DataFrame, or csv file\n")

    # for consistency swap rows and columns if data has the form experimentsXgenes
    if direction == "experiments":
        data_df = data_df.transpose()
        if isinstance(perts_df, pd.DataFrame):
            perts_df = perts_df.transpose()
    elif direction == "genes":
        pass
    else: 
        raise ValueError("Unknown direction alternative in snakeda. Supported designs are: genes and experiments")

    # sort data and pert data frames by experiment name for nicer plotting and correlation calculation of large datasets
    data_df = data_df.sort_index(axis=1)
    if perts_df != None:
        perts_df = perts_df.sort_index(axis=1)

    # make lookup dictionary for replicates
    if pert_design == 'multiple':
        rep_dict, norep = make_reps_lookup_dict(data_df, norep, 'multi_name', synthetic)
    elif pert_design == 'single':
        if isinstance(perts_df, pd.DataFrame):
            rep_dict, norep = make_reps_lookup_dict(perts_df, norep, 'single_pert') #does not need synthetic tag as the names are not used for dictionary creation
        else:
            rep_dict, norep = make_reps_lookup_dict(data_df, norep, 'single_name', synthetic)
    else:
        raise ValueError("Unknown pert_design type in snakeda. Supported designs are: single and multiple")

    # Make plots
    distribution_plot = make_histogram.plot(data_df.values, out) # histogram
    heatmap_plot = make_heatmap.plot(data_df, out) #heatmap
    exp_pca_plot = make_PCA.plot(data_df.values, rep_dict, out, "experiments") #PCA for experiments
    gene_pca_plot = make_PCA.plot(data_df.values, rep_dict, out, "genes") #PCA for genes

    # Calculate basic statistics for the data
    stats = {"Elements" : data_df.shape[0]*data_df.shape[1], # number of elements
        "Unique values" : len(np.unique(data_df.values)), # number of unique elements
        "Percent unique values" : round_significant(100 * len(np.unique(data_df.values))/(data_df.shape[0]*data_df.shape[1]), 2), # % unique
        "Zeros" : int(np.count_nonzero(data_df.values==0)), # number of zeros
        "Percent zeros" : round_significant(100 * np.count_nonzero(data_df.values==0)/(data_df.shape[0]*data_df.shape[1]), 2), # % zeros
        "Matrix shape" : str(data_df.shape[0]) + ", " + str(data_df.shape[1]) #shape of matrix
            }

    stats_temp = {"Mean" : round_significant(float(np.mean(data_df.values)), 4), # mean
        "Median" : round_significant(float(np.median(data_df.values)), 4), # median
        "Max" : round_significant(float(np.amax(data_df.values)), 4), # max
        "Min" : round_significant(float(np.amin(data_df.values)), 4) # min
            }

    stats.update(stats_temp)

    stats_temp = {"Variance" : round_significant(float(np.var(data_df.values)), 4), # variance
        "Standard deviation" : round_significant(float(np.std(data_df.values)), 4), # standard deviation
        "Skewness" : round_significant(float(scipy.stats.skew(data_df.values).mean()), 4), # skew
        "Signal to noise ratio" : round_significant(float(signal_to_noise_ratio(data_df, rep_dict, norep)), 4) # signal to noise ratio, calculated as mean/std
            }

    stats.update(stats_temp)

    # check so matrix is small enough to not run into typical memory problem
    if data_df.shape[0]<cutoff*2:
        svd = np.linalg.svd(data_df.values)[1] # singular value decomposition
        stats_temp = {"Condition number" : round_significant(float(np.linalg.cond(data_df.values)), 4), # condition number
                "Max of SVD" : round_significant(float(np.max(svd)), 4), # max of svd
                "Median of SVD" : round_significant(float(np.median(svd)), 4), # median of svd
                "Min of SVD" : round_significant(float(np.min(svd)), 4) # min of svd
                    }
    else:
        stats_temp = {"Condition number" : round_significant(float(np.linalg.cond(data_df.values)), 4), # condition number
                "Max of SVD" : np.nan, # max of svd
                "Median of SVD" : np.nan, # median of svd
                "Min of SVD" : np.nan # min of svd
                }

    stats.update(stats_temp)

    if pert_design == 'single':
        median_rank, median_p_rank, times_smallest, times_largest = pert_rank.calc(data_df, rep_dict)
        stats_temp = {"Median rank of perturbed gene" : median_rank, # median rank of perturbed gene in experiment
            "Median % rank pert gene" : round_significant(median_p_rank, 4)*100, # median percent rank of perturbed gene in experiment
            "Times lowest rank" : times_smallest, # number of times the perturbed gene is ranked lowest
            "Percent lowest rank" : round_significant(times_smallest/data_df.shape[1], 4)*100, # percent of times the perturbed gene is ranked lowest
            "Times highest rank" : times_largest, # number of times the perturbed gene is ranked highest
            "Percent highest rank" : round_significant(times_largest/data_df.shape[1], 2)*100 # percent of times the perturbed gene is ranked highest
                }
    else:
        # to ensure uniformity of output in spite of no replicates
        stats_temp = {"Median rank of perturbed gene" : "", 
            "Median % rank pert gene" : "",
            "Times lowest rank" : "",
            "Percent lowest rank" : "",
            "Times highest rank" : "",
            "Percent highest rank" : ""
                }

    stats.update(stats_temp)   

    if not norep: 
        if data_df.shape[1]<cutoff:
            # call function to calculate correlation statistics
            median_corr, median_rep_corr, num_greater_repcorrs = rep_corr_test.test(data_df, rep_dict)

            stats_temp = {"Median observation correlation" : round_significant(median_corr, 4), # median correlation of all columns
                "Median replicate correlation" : round_significant(median_rep_corr, 4), # median correlation of replicate groups
                "# replicate corr>median corr" : int(num_greater_repcorrs) # number of correlations within replicate groups greater than the overall median
                    }
        else:
            # call function to calculate correlation statistics ofr large data sets
            mean_corr, mean_rep_corr = rep_corr_test_lrg.test(data_df, rep_dict)

            stats_temp = {"Mean observation correlation" : round_significant(mean_corr, 4), # mean correlation of all columns
                "Mean replicate correlation" : round_significant(mean_rep_corr, 4), # mean correlation of replicate groups
                "# replicate corr>median corr" : "" # number of correlations within replicate groups greater than the overall median, not calculated on large data sets
                    }
        stats.update(stats_temp) 

        num_rep = [len(reps) for reps in rep_dict.values()]

        stats_temp = {"Median # of replicates" : round_significant(float(np.median(num_rep)), 2), # median number of replicates
            "Min # of replicates" : int(np.amin(num_rep)), # min number of replicates
            "Max # of replicates" : int(np.amax(num_rep)) # max number of replicates
                      }
    else:
        # to ensure uniformity of output in spite of no replicates
        stats_temp = {"Median observation correlation" : "",
            "Median replicate correlation" : "",
            "# replicate corr>median corr" : "",
            "Median # of replicates" : 1, # median number of replicates
            "Min # of replicates" : 1, # min number of replicates
            "Max # of replicates" : 1 # max number of replicates
                }

    stats.update(stats_temp)

    #output formatting
    if out:
        if not out.endswith(".png"):
            distribution_plot = format(distribution_plot.decode('utf-8'))
            heatmap_plot = format(heatmap_plot.decode('utf-8'))
            exp_pca_plot = format(exp_pca_plot.decode('utf-8'))
            gene_pca_plot = format(gene_pca_plot.decode('utf-8'))
    
    output_dict = {"Statistics": stats,
        "Plots":{
            "Histogram": distribution_plot,
            "Heatmap": heatmap_plot,
            "Experiment PCA": exp_pca_plot,
            "Gene PCA": gene_pca_plot
        }}
    
    if out:
        # if only json is desired, write full dictionary to file
        if out.endswith(".json"):
            with open(out, "w") as f:
                json.dump(output_dict, f, indent=4)
        # if only html is desired, input dictionary into to_html function
        elif out.endswith(".html"):
            json_to_html(out, output_dict)
        # if png is desired, make merged image of all figures, write statistics to json file
        elif out.endswith(".png"):
            distribution_img = Image.open(distribution_plot)
            heatmap_img = Image.open(heatmap_plot)
            exp_pca_img = Image.open(exp_pca_plot)
            gene_pca_img = Image.open(gene_pca_plot)

            out_img = Image.new("RGB", (2*distribution_img.size[0], 2*distribution_img.size[1]))

            out_img.paste(distribution_img, (0,0))
            out_img.paste(heatmap_img, (0, distribution_img.size[1]))
            out_img.paste(exp_pca_img, (distribution_img.size[0], 0))
            out_img.paste(gene_pca_img, (distribution_img.size[0], distribution_img.size[1]))

            out_img.save(out)

            with open(out[:-4]+".json", "w") as f:
                json.dump(output_dict["Statistics"], f, indent=4)
        # write full dictionary to json file and then reformat to html
        else:
            with open(out+".json", "w") as f:
                json.dump(output_dict, f, indent=4)
            json_to_html(out)
    else:
        # return statistics to standard output, figues will be shown one by one
        return output_dict["Statistics"]

# if script is called in terminal, parse terminal input
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for general data analysis of gene expression with one or more perturebed genes", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("data", help="the data to analyse, can be a pandas.DataFrame, GRNmodel or csv file with a matrix of genesXexperiments")
    parser.add_argument("-p", "--perts", dest="perts_df", default=None, help="optional seperate perturbation matrix as pandas.DataFrame or csv file, only used if pert_design is set to single")
    parser.add_argument("-nr", "--norep", dest="norep", default=False, type=bool,help="option for when there are no replicates. Will limit some analyses")
    parser.add_argument("-pd", "--pert_design", dest="pert_design", default="single", type=str, help="option to change between designs with a single and multiple perturbed genes per experiment")
    parser.add_argument("-o", "--out", dest="out", default=None, help="option to write output to file, name of file without file extention will create both json and html, can otherwise be specified to json, html or png")
    parser.add_argument("-d", "--direction", dest="direction", default="genes", help="option to specify if genes or experiments are on the rows")
    parser.add_argument("-s", "--synthetic", dest="synthetic", default=False, help="option for if the data has naming standard of synthetic data")

    # move input to args
    args = parser.parse_args()

    # run eda tool
    snakeda(args.data, args.perts_df, args.norep, args.pert_design, args.out, args.direction, args.synthetic)