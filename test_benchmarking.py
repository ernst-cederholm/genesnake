# script reads files of data and true network made in matlab's geneSPIDER
# it infers a network with lsco and provides performance statistics like F1-score, MCC etc.

import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from genesnake.benchmarking.compare_networks import compare_networks
from methods.grn_inference import infer_networks, compare_networks, data_summary
#from genesnake.benchmarking import compare_networks, data_summary


Anet = scipy.io.loadmat('C:\\Users\\mateu\\Desktop\\PostDocSU\\3p_geneSNAKE\\sample_networks_for_geneSNAKE\\Anet_g50_S3_snr01_rep2_1.mat')
Y = scipy.io.loadmat('C:\\Users\\mateu\\Desktop\\PostDocSU\\3p_geneSNAKE\\sample_networks_for_geneSNAKE\\y_g50_S3_snr01_rep2_1.mat')

Anet2 = pd.DataFrame.from_dict(Anet['outA'])
Y2 = -pd.DataFrame.from_dict(Y['outY'])

# sample P matrix (sP) 50x50
sP1 = np.eye(len(Y2), len(Y2))  # create matrix with 1's on diagonal (for 1 replicate)
sP1 = pd.DataFrame(sP1)
P = pd.concat([sP1, sP1], axis=1)  # as two replicates

data_summary(Y2, P)

# ls for inferring networks
zvec = np.logspace(-6, 0, 30)
aests = infer_networks(Y2, P, 'lsco', zvec)

out = compare_networks(aests, Anet2, True, False)

print(out['AUROC'])
# plot a boxplot with selected statistics
#fig = plt.figure()
#plt.boxplot(out[['mcc', 'f1', 'ppv', 'bm', 'fdr', 'auc']], labels=['MCC', 'F1', 'PPV', 'BM', 'FDR', "AUC"])
#plt.show()