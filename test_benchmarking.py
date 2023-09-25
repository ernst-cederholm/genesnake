# script reads files of data and true network made in matlab's geneSPIDER
# it infers a network with lsco and provides performance statistics like F1-score, MCC etc.

import scipy.io
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

#from genesnake.benchmarking.compare_networks import compare_networks
from methods.grn_inference import infer_networks, compare_networks, data_summary
#from genesnake.benchmarking import compare_networks, data_summary


Anet = scipy.io.loadmat('C:\\Users\\mateu\\Desktop\\PostDocSU\\3p_geneSNAKE\\sample_networks_for_geneSNAKE\\Anet_g50_S3_snr01_rep2_1.mat')
Y = scipy.io.loadmat('C:\\Users\\mateu\\Desktop\\PostDocSU\\3p_geneSNAKE\\sample_networks_for_geneSNAKE\\y_g50_S3_snr01_rep2_1.mat')

Anet2 = pd.DataFrame.from_dict(Anet['outA'])
sY = pd.DataFrame.from_dict(Y['outY'])

# sample P matrix (sP) 50x50
sP1 = np.eye(len(sY), len(sY))  # create matrix with 1's on diagonal (for 1 replicate)
sP1 = pd.DataFrame(sP1)
P = pd.concat([sP1, sP1], axis=1)  # as two replicates

from sklearn.svm import SVR
from sklearn.inspection import permutation_importance

svr_poly = SVR(kernel="poly")
aest = []
n = sY.shape[0]
for i in range(0, n):
    Y = sY.transpose()
    print(Y)
    svr_poly.fit(Y, Y.iloc[:,i])
    perm_importance = permutation_importance(svr_poly, Y, Y.iloc[:,i])
    print(perm_importance)

#data_summary(Y2, P)
#print(Y2)
# ls for inferring networks
#zvec = np.logspace(-10, -5, 30)
#zvec = np.linspace(-1, 2, 30)
#aests = infer_networks(Y2, P, 'rbfsvm', zvec)
#print(zvec)
#print(aests[0])
#out = compare_networks(aests, Anet2, False, False)

#print(out['AUROC'])
#print(max(out['metrics']['f1']))

# plot a boxplot with selected statistics
#fig = plt.figure()
#plt.boxplot(out[['mcc', 'f1', 'ppv', 'bm', 'fdr', 'auc']], labels=['MCC', 'F1', 'PPV', 'BM', 'FDR', "AUC"])
#plt.show()