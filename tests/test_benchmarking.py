# script reads files of data and true network made in matlab's geneSPIDER
# it infers a network with lsco and provides performance statistics like F1-score, MCC etc.

import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
from genesnake.benchmarking.compare_networks import compare_networks
from genesnake.benchmarking.cutoff_handler import cutoff_handler
from methods.grn_inference import infer_networks
#from genesnake.benchmarking import compare_networks, data_summary

# gold standard net
Anet = scipy.io.loadmat('C:\\Users\\mateu\\Desktop\\PostDocSU\\3p_geneSNAKE\\sample_networks_for_geneSNAKE\\nets_lscon_2rep_50gns_snr005.mat')
# data
Y = scipy.io.loadmat('C:\\Users\\mateu\\Desktop\\PostDocSU\\3p_geneSNAKE\\sample_networks_for_geneSNAKE\\Y_lscon_2rep_50gns_snr005.mat')

# select some net to fake gold standard and data
goldst = pd.DataFrame.from_dict(Anet['Aest'][...,25])
sY = pd.DataFrame.from_dict(Y['Y'])

# sample P matrix (sP) 50x50
sP1 = np.eye(len(sY), len(sY))  # create matrix with 1's on diagonal (for 1 replicate)
sP1 = pd.DataFrame(sP1)
P = pd.concat([sP1, sP1], axis=1)  # as two replicates

inetA = infer_networks(sY, P, "lscon")
inets = cutoff_handler(inetA) # create span of networks from full to empty, inetA is pandas data frame, inets is a list of pandas data frames
out = compare_networks(inets, goldst, exclude_diag=False, include_sign=False) # goldst is pandas data frame

# print all metrics as data frame
outm = out['metrics']
print(outm)

################ plotting results #####################
# plot a boxplot with selected statistics
figure, axis = plt.subplots(1, 2)
pal1 = ['#f62681','#00ffff','#fef250','#ffa500','#c8a2c8']
bp = axis[0].boxplot(outm[["mcc","f1","ppv","fdr","tpr"]], labels=['MCC', 'F1', 'PPV', 'FDR', 'TPR'], patch_artist=True)
axis[0].grid(axis='y')

for patch, color in zip(bp['boxes'], pal1):
    patch.set_facecolor(color)

for median in bp['medians']:
    median.set(color='black',
               linewidth=3)

data = {'AUROC': out['AUROC'], 'AUPR': out['AUPR']}
courses = list(data.keys())
values = list(data.values())

# creating the bar plot of AUPR and AUC
axis[1].bar(courses, values, color=['#9bbb59','#38761d'], width=0.4)

for n, ax in enumerate(axis):
    ax.text(-0.05, 1.05, string.ascii_uppercase[n], transform=ax.transAxes,
            size=18, weight='bold')

plt.show()