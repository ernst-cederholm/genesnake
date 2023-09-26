import pandas as pd
import numpy as np
import itertools

def make_new_ODEmodel(network,combines,names=None):
    """
    Function that builds an ODE model from an input network. 
    The function will build a model based on [ref] to describe the 
    probabalistic system of RNA transcription using RNA and protein concectraions.
    This will then be stored in a string object that can be executed using the exec command. 
    The string object can also be written to file and imported to python for later use or
    saved in a genesnake.model json file. 
    INPUT: 
        network - A numpy array or pandas dataframe of size genes*genes that describes interactions
                  between the genes such that regulator > target is row to column
        combines - A dictonary that describes the possible combinations between genes
                   such that the dict G0:[G1_G2:0,G2_G3:1] describes that G1 and G2 does not cooperate 
                   while gene G3 and G2 does cooperate.
        names -[optional]- A list of names of the genes in the network, should be in the same order as genes are seen in the 
                           network. NOTE: Must be used if network is a numpy array
    OUTPUT: 
        model - A string object which defines a function with equations that can be used in 
                an ODE solver to describe the system (Note should be feed to exec(model,global()))
    """

    # first a quick test to make sure input is correct
    if not isinstance(network, pd.DataFrame) and not isinstance(network, np.ndarray):
        raise TypeError("network must be of type numpy.ndarray or pandas.DataFrame.")
    elif isinstance(network, np.ndarray) and not names:
        raise ValueError("When using numpy.array the gene names must be provided in the names argument.") 
    if not isinstance(combines,dict):
        raise TypeError("combines must be a dictonary with key=gene value=the possible combinations of that gene. See genesnake.network.get_combinations for detailed format")

    # if network is pandas dataframe turn that in to a numpy.
    if isinstance(network, pd.DataFrame):
        # if names are not set seperatly assign the index as names 
        if not names:
            names = network.index.astype(str).to_list()
        # and set network to a numpy array
        network = network.values
        
    # to make sure nothing strange happens check that names and rows are the same number
    if not len(names)==network.shape[0]:
        raise ValueError("The number of given names seems to be different from the number of genes in the network")
    
    # set a varaible for the number of genes in the system
    genes = network.shape[0]
    
    # creat a list which we can use to apped the gene output in
    # so that they come in the exact same order as they did as input
    out = [None]*(2*genes)

    # remove any . in the names in combinations as . can not be used in variable names
    combines = {
    str(key).replace(".", ""): [str(v).replace(".","") for v in value] for key, value in combines.items()
    }
    # same for names
    names = [str(i).replace(".","") for i in names]
    
    # Define a number of inital values and other small things 
    model = """def model(t,y,alphas,mis,ris,ns,ks,lambda_RNA,lambda_P,comb_mat,pert):    
    # Auto generated function for describing a system of ODE equations 
    # that describes a GRN"""

    # NOTE that model creation is done in 2 loops to make sure all variables are defined at the start of the
    # function before any actual code. due to this some parts of the code is duplicated here  
    # get all parameters from the input
    # NOTE the strange formating of the string objects is to make sure the tab spacing inside the string object is correct
    # for when executing this as a function 
    for i in range(genes):
        # get the key name to match with combinations and so on 
        g = names[i]
        # set variables that's all the same for gene g
        model+="""
    RNA_{}=y[{}]""".format(g,i)
        model+="""
    prot_{}=y[{}]""".format(g,genes+i)
        model+="""
    mi_{}=round(max(mis[{}]*(1+pert[{}]),0),4) #take max of input and 0 to ensure that the expression is never <0""".format(g,i,i)
        model+="""
    ri_{}=ris[{}]""".format(g,i)
        model+="""
    n_{}=ns[{}]""".format(g,i)
        model+="""
    lambdaRNA_{}=lambda_RNA[{}]""".format(g,i)
        model+="""
    lambdaP_{}=lambda_P[{}]""".format(g,i)
        model+="""
    alpha_{}=alphas[\"{}\"]""".format(g,g)
        # get all relevant alphas for this gene:

        # get the genes that regulates the current genes 
        inds = np.where(network[:,i])[0]
        # get the combinations of regulators that are avalible for this gene
        # This part should be rewritten to handle a None input of comb_mat
        gene_combs = [name.split(":")[0] for name in combines[g] if int(name.split(":")[1])>0]
        print(gene_combs)
        
        # get the regulators for the current gene as well as the combinations of regulators
        if len(inds) > 0:
            regs = [names[ind] for ind in inds]
            regs = regs+gene_combs
        else:
            regs = []
        print(regs)
        # get an alpha for each regulation    
        for gene in regs:
            model+="""
    alpha_{}_{}=alphas[\"{}_{}\"]""".format(g,gene,g,gene)
            model+="""
    k_{}_{}=ks[\"{}_{}\"]""".format(g,gene,g,gene)

        # add a white space
        model+="""
        """
    # loop over all genes and capture the describing features 
    for i in range(genes):
        # get the key name to match with combinations and so on 
        g = names[i]

        # get the genes that regulates the current genes 
        inds = np.where(network[:,i])[0]

        # transform the network in to a lookup table we can use to
        # get the sign of the interaction
        signs = dict()
        for j in inds:
            signs[names[j]] = network[j,i]

        # get the combinations of regulators that are avalible for this gene
        # This part should be rewritten to handle a None input of comb_mat
        gene_combs = {name.split(":")[0]:int(name.split(":")[1]) for name in combines[g]}

        # get the regulators for the current gene as well as the combinations of regulators
        if len(inds) > 0:
            regs = [names[ind] for ind in inds]
            regs = regs+list(gene_combs.keys())
        else:
            regs = []

        # once all needed information is avalible start to go over each regulator (and combinations)
        if len(regs) > 0:
            # start an equation that will describe the interactions
            equation = "d_{} = mi_{}*".format(g,g)        
            # along with a numerator and denominator for the final equation 
            top=[]
            bottom = ["1"]
            # add the description for each gene or combination of genes 
            for gene in regs:
                # if there is more than one gene in the input we take the sign of the
                # first gene. which gene here does not matter as they should have the same
                # sign to colaberate. 
                if "_" in gene:
                    
                    # don't add 0 if there is no collaboration as it's just a massive equation*0
                    if gene_combs[gene]==0:
                        continue
                    
                    split_coop_gene = gene.split("_")
                    #sign = signs[split_coop_gene[0]]
                    
                    ans_top = ["{}*alpha_{}_{}".format(float(gene_combs[gene]),g,gene)]
                    ans_bottom = []

                    #if sign > 0:
                    for q in split_coop_gene:
                        ans_top.append("((prot_{}/k_{}_{})**n_{})".format(q,g,gene,g))
                        ans_bottom.append("((prot_{}/k_{}_{})**n_{})".format(q,g,gene,g))
                    #else:
                    #    for q in split_coop_gene:
                    #        ans_top.append("((k_{}_{}/prot_{})**n_{})".format(g,gene,q,g))
                    #        ans_bottom.append("((k_{}_{}/prot_{})**n_{})".format(g,gene,q,g))

                    top.append("("+"*".join(ans_top)+")")
                    bottom.append("("+"*".join(ans_bottom)+")")
                else:
                    #sign = signs[gene]
                    #if sign > 0:
                    top.append("(alpha_{}_{}*(prot_{}/k_{}_{})**n_{})".format(g,gene,gene,g,gene,g))
                    bottom.append("((prot_{}/k_{}_{})**n_{})".format(gene,g,gene,g))
                    #elif sign < 0:
                    #    top.append("(alpha_{}_{}*(k_{}_{}/prot_{})**n_{})".format(g,gene,g,gene,gene,g))
                    #    bottom.append("((k_{}_{}/prot_{})**n_{})".format(g,gene,gene,g))
                        
            # and finally join these togheter in to a single equation that describes the cahnge in gene expression
            equation+="(("+"alpha_{}".format(g)+"+"+"+".join(top)+")/("+"+".join(bottom)+"))-lambdaRNA_{}*RNA_{}".format(g,g)
            # add a second equation that describes the change in protein 
            prot = """dp_{} = ri_{}*RNA_{}-lambdaP_{}*prot_{}""".format(g,g,g,g,g)
        # if there are no regulators we add mi*0.5 as expressiob to simulate an unknown effect of about half maximum transcription 
        else:
            equation = "d_{} = mi_{}*0.5-(lambdaRNA_{}*RNA_{})".format(g,g,g,g)
            prot = """dp_{} = ri_{}*RNA_{}-lambdaP_{}*prot_{}""".format(g,g,g,g,g)
            
            
        # add the newly created dG and dp (change in RNA and protein) in the order they are found in the input
        # to the output variable 
        out[i] = "d_{}".format(g)
        out[genes+i] = "dp_{}".format(g)
        
        # add the gene expression and protein change equations to the model it self
        model+="""
    {}""".format(equation)
        model+="""
    {}
        """.format(prot)
        
    # once all genes are added create a list object with all outputs
    # by joining the list out in to a string representation of a list
    out = ",".join(out)
    
    # and add this to the model along with a return statement for this list 
    model+="""
    step = [{}]
    return step""".format(out)

    return model
