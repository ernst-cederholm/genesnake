import numpy as np
import statistics as st
import math as mt
import scipy

def zinbme(data, prob_zi, prob_nb, snr):
    zi = np.random.choice([0, 1], p=[prob_zi, 1-prob_zi], size=len(data), replace=True)  # inflation with zero
    nb = np.random.negative_binomial(1, prob_nb, size=len(data))  # negative binomial distribution
    #nb = scipy.stats.nbinom.rvs(2, prob_nb, size=len(data))
    zinb = np.multiply(zi, nb)
    m = st.sqrt(st.variance(data) / snr*(st.variance(nb)))  # scaling factor to adjust noise values to data
    return zinb*m

data_counts =  [2, 1, 3, 5, 10, 2, 5, 0, 1, 2, 1, 7, 3, 5, 10, 2, 5, 0, 1, 10] # X matrix
print(data_counts)
# E - noise
Ezinb = zinbme(data_counts, 0.01, 0.5, 1)  # params: data in counts, dropout probability (1 means all dropouts), correctness probability (1 means no errors)
print(Ezinb)
Y = data_counts + Ezinb
print(Y)
SNR = st.variance(data_counts) / (st.variance(Ezinb))  # (variance X) / (variance E)
print(SNR)