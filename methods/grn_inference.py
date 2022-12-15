# MG
# load libraries

import numpy as np
import pandas as pd
import sys
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

# sample Y matrix (sY) 4x4 for 1 replicate
# 1st replicate
sY1 = pd.DataFrame({"G1": [-1, 0, -0.65, 0],
                    "G2": [-0.2, -0.98, 0, 0.01],
                    "G3": [-0.1, 0.09, -1, 0.2],
                    "G4": [0, -0.09, 0.1, -0.96]})
# 2nd replicate
sY2 = pd.DataFrame({"G1_2": [-0.9, 0.1, -0.7, 0],
                    "G2_2": [-0.05, -1, 0, 0.4],
                    "G3_2": [0, -0.04, -0.88, 0.6],
                    "G4_2": [0, 0.02, 0.3, -0.99]})
# merge replicates
sY = pd.concat([sY1, sY2], axis=1)

# sample P matrix (sP) 4x8
sP1 = np.eye(4, 4)  # create matrix with 1's on diagonal (for 1 replicate)
sP1 = pd.DataFrame(sP1)
sP = -pd.concat([sP1, sP1], axis=1)  # as two replicates


# make N networks under various cutoffs using vector of zeta values as thresholds
# auto means that it takes 100 thresholds between 0 and maximum absolute value from the full network
def handle_zeta(A, zvec, zvec_auto=False):
    aests = []
    if (zvec_auto):
        zvec = np.linspace(0, max(abs(A)), 100)
    else:
        ep = sys.float_info.epsilon
        zmin = min(abs(A[A > 0])) - ep
        zmax = max(abs(A[A > 0])) + ep * 10
        zvec = zvec * (zmax - zmin) + zmin
    for z in range(1, len(zvec)):
        Atmp = A
        Atmp[abs(Atmp) <= zvec[z]] = 0
        aests.append(round(Atmp, ndigits=5))
    return aests


# function for inferring networks
def grn_inference(Y, P, method, zvec):
    if method == "lsco":  # least squares
        aest = np.matmul(-P, np.linalg.pinv(Y))
        aests = handle_zeta(aest, zvec, False)
    if method == "lscon":  # normalized least squares
        aest = np.matmul(-P, np.linalg.pinv(Y))
        aest = pd.DataFrame(aest)
        for column in aest:
            aest[column] = np.divide(aest[column], sum(abs(aest[column]))) * Y.shape[0]
        aests = handle_zeta(aest, zvec, False)
    if method == "zscore":  # Z-score
        outNet = []
        P2 = P.transpose()
        for i in P2:
            inds = np.nonzero(abs(P.iloc[i].array))
            Y2 = Y.transpose()
            tmpAest = np.divide((np.mean(Y2.iloc[inds], axis=0) - np.mean(Y, axis=1)), np.std(Y, axis=1))
            outNet.append(tmpAest)
        aest = pd.DataFrame(outNet).transpose()
        aests = handle_zeta(aest, zvec, False)
    if method == "lasso":  # lasso
        aests = []
        for z in zvec:
            lasso_mod = Lasso(alpha=z)
            out_net_tmp = []
            for i in P.transpose():
                lasso_mod.fit(Y.transpose(), -P.iloc[i])
                out_net_tmp.append(lasso_mod.coef_)
            out_net_tmp = pd.DataFrame(out_net_tmp)
            aests.append(out_net_tmp)
    if method == "elnet":  # elastic net
        aests = []
        for z in zvec:
            lassoMod = ElasticNet(alpha=z, random_state=0)
            outNetTmp = []
            for i in P.transpose():
                lassoMod.fit(Y.transpose(), -P.iloc[i])
                outNetTmp.append(lassoMod.coef_)
            outNetTmp = pd.DataFrame(outNetTmp)
            aests.append(outNetTmp)
    return aests


# testing the GRN inference
zvec = np.logspace(-6, 0, 30)
Anets = grn_inference(sY, sP, "lasso", zvec)

# simple visualization of network
# select n-th network and display it
n = 21
print(Anets[n])
net_to_show = pd.DataFrame(Anets[n])

G = nx.to_networkx_graph(net_to_show.transpose(), create_using=nx.DiGraph()) # directed network
G.remove_edges_from(nx.selfloop_edges(G))  # remove selfloops
pos = nx.circular_layout(G) # set up layout
wgs = nx.get_edge_attributes(G, "weight") # give it a weight
# draw the network with visible labels, opacity 50% and arrows
nx.draw(G, pos, with_labels=True, node_size=1000, alpha=0.5, font_weight="bold", arrows=True, arrowstyle='simple', node_color='gray')
plt.axis('on')
plt.show()
