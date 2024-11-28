import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from methods.grn_inference import infer_networks, cutoff_handler

X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

y[::5] += 3 * (0.5 - np.random.rand(8))

svr_rbf = SVR(kernel="rbf")
svr_lin = SVR(kernel="linear")
svr_poly = SVR(kernel="poly")


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

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

aest = []
n = sY.shape[0]
for i in range(0, n):
    print(i)
    # create a regressor object
    regressor = DecisionTreeRegressor(random_state=0)
    Y = sY.transpose()

    #print(Y.drop(i, inplace=False, axis=1))
    #print(Y.iloc[:,i])
    #X_train, X_test, y_train, y_test = train_test_split(Y.drop(i, inplace=False, axis=1), Y.iloc[:,i], test_size=n, random_state=0)
    fitc = regressor.fit(Y, Y.iloc[:,i])
    aest.append(fitc.feature_importances_)
print(pd.DataFrame(aest))
    #pred = fitc.predict(X_test)
    #print(pred)

#aest2 = pd.DataFrame(np.concatenate(aests))
#print(aest2)
#aests2 = cutoff_handler(pd.DataFrame(aests), 0.005, False)
#print(aests2)
from sklearn.inspection import permutation_importance

aest = []
n = sY.shape[0]
for i in range(0, n):
    Y = sY.transpose()
    svr_poly.fit(Y, Y.iloc[:,i])
    perm_importance = permutation_importance(svr_poly, Y, Y.iloc[:,i])
    print(perm_importance['importances_mean'])

    #aest.append(fitc.feature_importances_)
#print(pd.DataFrame(aest))
