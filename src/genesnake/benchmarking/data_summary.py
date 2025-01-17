import numpy as np
from scipy.stats import shapiro, kstest
import statistics as st
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

def data_summary(data, p_matrix):
	"""
	Function for displaying data summary.
	INPUT:
		data - Y matrix
	"""

	nrw = len(data.index)
	ncl = len(data.columns)

	data2 = data.to_numpy()
	test_nor = kstest(data2.flatten(), 'norm')

	desc = list(['0 values', 'Inf values', 'NaN values', 'positive values', 'negative values'])
	vals = np.asarray([np.isclose(data, 0.0).sum(),
					   np.isinf(data).sum().sum(),
					   data.isnull().values.sum(),
					   (data > 0).values.sum(),
					   (data < 0).values.sum()]) / (nrw * ncl) * 100

	# perturbation ranking
	# for i in range(0, p_matrix.shape[1]-1):
	#    print(np.array(p_matrix.ix[:,i]))

	pert_rank = []

	for i, column in enumerate(p_matrix):
		ind_p = p_matrix.iloc[:, i] == 1
		tmp_data_col = data[i]
		pert = abs(np.array(tmp_data_col[np.invert(ind_p)])) > abs(np.array(tmp_data_col[ind_p]))
		# print(sum(pert == True))
		# print(type(pert))
		tmp_sum = sum(pert == True).tolist()
		# print((tmp_sum[0]))
		pert_rank.append(tmp_sum)

	# For Sine Function

	figure, axes = plt.subplots(2, 2, figsize=(12, 12))
	sns.kdeplot(np.array(data).flatten(), ax=axes[0, 0])
	axes[0, 0].title.set_text('Data density')
	sm.qqplot(data, line='45', ax=axes[0, 1])
	axes[0, 1].text(-1.5, 1.5, 'KS P value = ' + str(round(test_nor.pvalue, ndigits=2)), fontsize=12)
	axes[0, 1].title.set_text('Data normality')
	axes[1, 0].bar(desc, vals, color='slategrey', width=0.4)
	axes[1, 0].title.set_text('Values in the data')
	axes[1, 0].set_ylabel("Amount [%]")
	axes[1, 0].set_xticklabels(desc, rotation=45)
	# Combine all the operations and display
	sns.histplot(pert_rank, ax=axes[1, 1], kde=True, element="step")
	axes[1, 1].title.set_text('Perturbation ranking')
	plt.show()
