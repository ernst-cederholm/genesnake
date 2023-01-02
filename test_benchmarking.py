# script reads files of data and true network made in matlab's geneSPIDER
# it infers a network with lsco and provides statistics like F1-score, MCC etc.

import scipy.io
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
#from genesnake.benchmarking.compare_networks import compare_networks
from methods.grn_inference import infer_networks, cutoff_handler, compare_networks

Anet = scipy.io.loadmat('C:\\Users\\mateu\\Desktop\\sample_networks\\Anet_g50_S3_snr01_rep2_1.mat')
Y = scipy.io.loadmat('C:\\Users\\mateu\\Desktop\\sample_networks\\y_g50_S3_snr01_rep2_1.mat')

Anet2 = pd.DataFrame.from_dict(Anet['outA'])
Y2 = pd.DataFrame.from_dict(Y['outY'])

# sample P matrix (sP) 50x50
sP1 = np.eye(len(Y2), len(Y2))  # create matrix with 1's on diagonal (for 1 replicate)
sP1 = pd.DataFrame(sP1)
P = pd.concat([sP1, sP1], axis=1)  # as two replicates



# ls for inferring networks
zvec = np.logspace(-6, 0, 30)
aests = infer_networks(Y2, P, 'lasso', zvec)
print(aests)


out = compare_networks(aests, Anet2, True, True)

fig = plt.figure()
plt.boxplot(out['f1_score'])
plt.show()