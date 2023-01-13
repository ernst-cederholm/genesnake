import numpy as np
from scipy.stats import shapiro, kstest
import statistics as st
def data_summary(data):
    """
    Function for displaying data summary.
    INPUT:
        data - Y matrix
    """
    print('### some statistics for your data ####')
    nrw = len(data.index)
    ncl = len(data.columns)
    print('Data has', nrw, 'genes (rows)', 'and', ncl, 'samples (columns)')

    if (ncl >= 50):
        data2 = data.to_numpy()
        test_nor = kstest(data2.flatten(), 'norm')  # >= 50
    else:
        test_nor = shapiro(data)  # < 50

    if (test_nor.pvalue < 0.05):
        print('Data is not distributed normally')
    else:
        print('Data is distributed normally')


    print( '▶ min:', min(data), '▶ median: ', st.median(data), '▶ mean: ', st.mean(data), 'max:', max(data))

    print('Number of 0 values', np.isclose(data, 0.0).sum(), '[', round(np.isclose(data, 0.0).sum() / data.size * 100, 2), '%]')
    print('Number of Inf values', np.isinf(data).sum().sum(), '[', round(np.isinf(data).sum().sum() / data.size * 100, 2), '%]')
    print('Number of NaN values', data.isnull().values.sum(), '[', round(data.isnull().values.sum() / data.size * 100, 2), '%]')
    print('Number of positive values', (data > 0).values.sum(), '[', round((data > 0).values.sum() / data.size * 100, 2), '%]')
    print('Number of negative values', (data < 0).values.sum(), '[', round((data < 0).values.sum() / data.size * 100, 2), '%]')