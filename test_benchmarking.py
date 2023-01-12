# script reads files of data and true network made in matlab's geneSPIDER
# it infers a network with lsco and provides performance statistics like F1-score, MCC etc.

import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as st
#from genesnake.benchmarking.compare_networks import compare_networks
from methods.grn_inference import infer_networks, cutoff_handler, compare_networks

from scipy.stats import shapiro, kstest


Anet = scipy.io.loadmat('C:\\Users\\mateu\\Desktop\\PostDocSU\\3p_geneSNAKE\\sample_networks_for_geneSNAKE\\Anet_g50_S3_snr01_rep2_1.mat')
Y = scipy.io.loadmat('C:\\Users\\mateu\\Desktop\\PostDocSU\\3p_geneSNAKE\\sample_networks_for_geneSNAKE\\y_g50_S3_snr01_rep2_1.mat')

Anet2 = pd.DataFrame.from_dict(Anet['outA'])
Y2 = pd.DataFrame.from_dict(Y['outY'])
####
print('### some statistics for your data ####')
nrw = len(Y2.index)
ncl = len(Y2.columns)
print('Data has', nrw, 'genes (rows)', 'and', ncl, 'samples (columns)')

if (ncl >= 50):
    Y3 = Y2.to_numpy()
    test_nor = kstest(Y3.flatten(), 'norm')  # < 50
else:
    test_nor = shapiro(Y2)  # < 50

if (test_nor.pvalue < 0.05):
    print('Data is not distributed normally')
else:
    print('Data is distributed normally')


print('max:', max(Y2), 'min:', min(Y2), 'median: ', st.median(Y2), 'mean: ', st.mean(Y2))
####

# sample P matrix (sP) 50x50
sP1 = np.eye(len(Y2), len(Y2))  # create matrix with 1's on diagonal (for 1 replicate)
sP1 = pd.DataFrame(sP1)
P = pd.concat([sP1, sP1], axis=1)  # as two replicates

# ls for inferring networks
zvec = np.logspace(-6, 0, 30)
aests = infer_networks(Y2, P, 'lasso', zvec)

out = compare_networks(aests, Anet2, True, True)
print(out)
# plot a boxplot with selected statistics
fig = plt.figure()
plt.boxplot(out[['mcc', 'f1', 'ppv', 'bm', 'fdr']], labels=['MCC', 'F1', 'PPV', 'BM', 'FDR'])
plt.show()