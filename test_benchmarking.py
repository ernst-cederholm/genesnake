# script reads files of data and true network made in matlab's geneSPIDER
# it infers a network with lsco and provides performance statistics like F1-score, MCC etc.

import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
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


zvec = np.logspace(-5, 0, 100)

nets = infer_networks(sY, P, "zscore", zvec)
#print(nets)

out = compare_networks(nets, Anet2, exclude_diagonal=False, signed=False)

#print(out['AUPR'])
#print(out['AUROC'])
#print(max(out['metrics']['f1']))
#print(out)
#from sklearn.svm import SVR
#from sklearn.inspection import permutation_importance

#svr_poly = SVR(kernel="poly")
#aest = []
#n = sY.shape[0]
#for i in range(0, n):
#    Y = sY.transpose()
#    print(Y)
#    svr_poly.fit(Y, Y.iloc[:,i])
#    perm_importance = permutation_importance(svr_poly, Y, Y.iloc[:,i])
#    print(perm_importance)

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
outm = out['metrics']
print(outm)

# plot a boxplot with selected statistics
figure, axis = plt.subplots(1, 2)
pal1 = ['#f62681','#00ffff','#fef250','#ffa500','#c8a2c8']
bp = axis[0].boxplot(outm[["mcc","f1","ppv","fdr","tpr"]], labels=['MCC', 'F1', 'PPV', 'FDR', 'TPR'], patch_artist=True)
axis[0].grid(axis='y')

for patch, color in zip(bp['boxes'], pal1):
    patch.set_facecolor(color)

for median in bp['medians']:
    median.set(color ='black',
               linewidth = 3)


data = {'AUROC': out['AUROC'], 'AUPR': out['AUPR']}
courses = list(data.keys())
values = list(data.values())



# creating the bar plot
axis[1].bar(courses, values, color=['#9bbb59','#38761d'], width=0.4)

for n, ax in enumerate(axis):
    ax.text(-0.05, 1.05, string.ascii_uppercase[n], transform=ax.transAxes,
            size=18, weight='bold')

plt.show()