"""
Script that details how to run a simple geneSNAKE simulation 
"""

# first import some packages used for this script
# this is not needed to run geneSNAKE though geneSNAKE do make use of the
# same packages 
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
import numpy as np

# import a method for network inference to test the benchmarking
from methods.grn_inference import infer_networks


# Define some settings that are needed to run the example 
gene_number = 10 # how many genes the GRN should contain 
use_static_network = True # should the network be selected from one of two existing predefined networks
plot_results = True # should the final network, data and fold change be plotted at the end 
load_network = False # a network to load, replace False with path to file containing the network
print_params = False # option to print all parameters to terminal, exluding time points 
set_params = False # option to set all parameters to predefined values 


### actual code for using geneSNAKE starts here ###

# Import that genesnake package. Note this assumes the genesnake folder is in your python
# dist-packages folder or the same folder as this script. 
import genesnake as gs

# if using a static network select from a 5 or 10 gene network
# This will result in a network defined in the numpy matrix here and is
# usefull for testing what the different parameters have for effect as it will be
# a static network.
if use_static_network:
    genes = 5
    network = np.array([[0,0,0,0,0],[-1,0,0,0,0],[0,0,0,1,0],[0,1,0,0,0],[0,0,0,1,0]])
    network = pd.DataFrame(network)
    network.columns=["G"+str(i+1) for i in range(genes)]
    network.index = network.columns
# a network can also be loaded using the gs.grn.load_network functon to easily load a csv file in to a genesnake friendly format
# can also be easily done with pd.load_csv(file)
elif load_netowork:
    network = gs.grn.load_network(load_network,net_type="edgelist",sep=",",header="True",index=False)
# or created using one of the algorithms in gs.grn.func
# currently gs.grn.make_DAG, gs.grn.make_random and gs.grn.make_FFLATnetwork can be used for this
else:
    genes = gene_number
    # for DAG and random the syntax is 
    network = gs.grn.make_DAG(genes,1.2,sign_bias=0.6)
    # and for FFLatt it is
    #network = gs.grn.make_FFLATnetwork(NETWORK_SIZE=genes,FFL_ENRICHED=0,self_loops=False)
    
    
# Once a network is availeble make a ODE model and set parameters for it
# note the usage of dicts for feeding parameters options in to the generators.
# e.g combinations={"comb_prob":0.5} sets the probability of two genes have in collaborative regulatory interacton to 50/50 if
# a collbaoration is possible 
# This is done using:  
M = gs.GRNmodel.make_model(network,combinations={"comb_prob":0.5})

# the created model is both stored as a string object and added as a running function to the global namespace.
# to check the model one can use:
print(M.model)

# all parameters can also be set manually
# the following is an example to set all values for the parameters to a single value
if set_params:
    for key in M.parameters.alphas.keys():
        # for inhibition to work the alpha value for any inhibitor must always be 0  
        if M.parameters.alphas[key]!=0 and len(key.split("_"))>1:
            M.parameters.alphas[key]=1
        elif M.parameters.alphas[key]!=0:
            M.parameters.alphas[key]=0.1
            
        # the option for other parameters are a bit more flexible but setting them
        # to high or low may cause issues in the simulation 
        M.parameters.hill_coef = [4 for _ in range(genes)]
        M.parameters.dis_const = {key:0.8 for key in M.parameters.dis_const.keys()}
        M.parameters.lambda_RNA = [0.8 for _ in range(genes)]
        M.parameters.lambda_prot = [0.6 for _ in range(genes)]

# to check what parameters where set they can be printed from
# GRNmodel.parameters  
if print_params:
    for i in vars(M.parameters).keys():
        # Note that here time is not printed as this contains the time points for the simulation which by default are 10,000 values
        if i != "time":
            print(i,vars(M.parameters)[i])

# Once all the parameters is set a pertrubation matrix should  be created
# which can be easily done through the GRNmodel.set_pert function 
M.set_pert("diag",effect=(0.89,0.91),noise=0,multipert=3,exp=1,reps=3,sign=-1)

# to check the perturbation one can use
print(M.perturbation)

# With parameters and pertrubation in place the simulation can now be ran using the simulate_data function
# this also takes arguments for the noise to be added 
M.simulate_data(exp_type="ss",SNR=5,noise_model="normal")

# The save/load part and benchmarking part is still under construction and needs further testing 

# to save the model so that it can be reused there is a save function:
M.save("genesnake_test_model.pickle",kind="pickle")

# this model can then be loaded using
M2 = gs.GRNmodel.load("genesnake_test_model.pickle",kind="pickle")

### benchmarking ### 

# Currently there are not function in genesnake to calculate change e.g fold change but it can be
# easily done by the following snippet 
fold_change_data = np.zeros((genes,genes*3))
for i in range(genes*3):
    for j in range(genes):
        fold_change_data[j,i] = np.log2(M.data.values[j,i]/M.steady_state_RNA[j])
fold_change_data = pd.DataFrame(fold_change_data,index=M.data.index,columns=M.data.columns)


# once a simulation is completed and inference is performed genesnake can also be used to benchmark the results
# run some form of inference 
results = infer_networks(fold_change_data,M.perturbation,"lscon",np.logspace(-6,0,100))
# and load this in to the benchmarking function 
benchmark = gs.benchmarking.compare_networks(results,M.network,exclude_diagonal=True)
print("\n The LSCON method peformance are; AUPR:"+str(benchmark["AUPR"])+", AUROC:"+str(benchmark["AUROC"]))


# Currently there are not function in genesnake to calculate change e.g fold change but it can be
# easily done by the following snippet 
fold_change_data = np.zeros((genes,genes*3))
for i in range(genes*3):
    for j in range(genes):
        fold_change_data[j,i] = np.log2(M.noise_free_data.values[j,i]/M.steady_state_RNA[j])
fold_change_data = pd.DataFrame(fold_change_data,index=M.data.index,columns=M.data.columns)


### a simple way to look at the dynamice ### 

if plot_results:
    # draw a figure that shows the change from control to steady state

    fig = plt.figure(constrained_layout=True)
    (subfig1,subfig2, subfig3,subfig4) = fig.subfigures(4, 1)

    ax1 = subfig1.subplots(1,1)
    ax4 = subfig2.subplots(1,1)
    ax2 = subfig3.subplots(1,1)
    ax3 = subfig4.subplots(1,1)

    G = nx.from_pandas_adjacency(network,create_using=nx.DiGraph)
    edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
    colors = list()
    for w in weights:
        if w < 0:
            colors.append("r")
        else:
            colors.append("b")
    try:
        nx.draw_planar(G,with_labels=True,ax=ax1,edgelist=edges,edge_color=colors)
    except nx.NetworkXException: 
        nx.draw_spring(G,with_labels=True,ax=ax1,edgelist=edges,edge_color=colors)

    if genes <= 20:
        sns.heatmap(pd.DataFrame(M.steady_state_RNA).T,cmap="vlag",linecolor="k",linewidths=0.01,ax=ax4,annot=True)
        sns.heatmap(M.noise_free_data.iloc[:,0:genes],cmap="vlag",linecolor="k",linewidths=0.01,ax=ax2,annot=True)
        sns.heatmap(fold_change_data.iloc[:,0:genes],cmap="vlag",linecolor="k",linewidths=0.01,ax=ax3,annot=True)
    else:
        sns.heatmap(M.steady_state_RNA,cmap="vlag",linecolor="k",linewidths=0.01,ax=ax4,annot=True)
        sns.heatmap(M.noise_free_data.iloc[:,0:genes],cmap="vlag",linecolor="k",linewidths=0.01,ax=ax2)
        sns.heatmap(fold_change_data.iloc[:,0:genes],cmap="vlag",linecolor="k",linewidths=0.01,ax=ax3)
        
    subfig1.suptitle("GRN")
    subfig2.suptitle("inital steady_state_RNA")
    subfig3.suptitle("Noise free data")
    subfig4.suptitle("Noise free fold change")
    
    plt.show()


### condensed use case ###

# set number of genes 
genes = gene_number
# generate network  
network = gs.grn.make_DAG(genes,1.2,sign_bias=0.6)
# make a model for simulation
M = gs.GRNmodel.make_model(network,combinations={"comb_prob":0.5})
M.set_pert("diag",effect=(0.89,0.91),noise=0,multipert=3,exp=1,reps=3,sign=-1)
# run simulatons
M.simulate_data(exp_type="ss",SNR=5,noise_model="normal")
#and save model
M.save("genesnake_test_model.pickle",kind="pickle")
# save data and network as csv files
M.data.to_csv("genesnake_test_data.csv")
M.network.to_csv("genesnake_test_network.csv")

print("\n\nNOTE: this script created the files; genesnake_test_model.pickle,genesnake_test_data.csv and genesnake_test_network.csv.\nTo remove them use:\n\trm genesnake_test_model.pickle genesnake_test_data.csv genesnake_test_network.csv\n")
