import statistics

import numpy as np
import pandas as pd
import sys
from sklearn.linear_model import ElasticNet
from sklearn import svm
#from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from scipy.stats import shapiro, kstest
import statistics as st
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from statistics import mean
from sklearn.svm import SVR
from copy import deepcopy



def infer_networks(
		Y, P, 
		method, 
		zvec = np.logspace(-5, 0, 100),
		**kwargs):
	"""
	Function for inferring networks with various methods
	INPUT:
		Y - gene expressions, where columns are experiments and rows are genes
		P - binary matrix of perturbations corresponding to Y
		method - inference method to apply
		zvec - vector of zeta values for cutoff
	"""

	# Avoid modifying inputs. This happened before since the indices
	# from P and Y are copied to new data structures, but then that
	# goes by reference not copy in an unintuitive way.
	# Changing something about the network coming out of this function
	# should not reasonably modify Y and P.
	P = deepcopy(P)
	Y = deepcopy(Y)

	if method == "lsco":  # least squares, works OK
		P = P.T
		Y = Y.T
		from threadpoolctl import threadpool_limits
		with threadpool_limits(limits = 5):
			aest = np.matmul(-P, np.linalg.pinv(Y))
		aest.index = P.index
		aest.columns = P.index
		return aest.T
	if method == "lscon":  # normalized least squares, works OK
		P = P.T
		Y = Y.T
		from threadpoolctl import threadpool_limits
		with threadpool_limits(limits = 5):
			aest = np.matmul(-P, np.linalg.pinv(Y))
			aest = pd.DataFrame(aest)
			for column in aest:
				aest[column] = np.divide(aest[column], sum(abs(aest[column]))) * Y.shape[0]
		aest.index = P.index
		aest.columns = P.index
		return aest.T
	if method == "zscore_ab":
		"""
		Most of notation and code according to the zscore method from 
		the DREAM 3 paper. Search for z-score in the paper to find 
		the section.

		Conventions used for this method:
		Data: [experiments/cells/observations x genes/variables]
		Output network: [regulators x targets]

		Returns NaN for genes that are not perturbed in any experiments.

		DREAM 3 paper:
			Link
				https://doi.org/10.1371/journal.pone.0009202
			Citation
				Prill RJ, Marbach D, Saez-Rodriguez J, Sorger PK, Alexopoulos LG, Xue X, Clarke ND, Altan-Bonnet G, Stolovitzky G. Towards a rigorous assessment of systems biology models: the DREAM3 challenges. PLoS One. 2010 Feb 23;5(2):e9202. doi: 10.1371/journal.pone.0009202. Erratum in: PLoS One. 2010;5(3). doi: 10.1371/annotation/f633213a-dc4f-4bee-b6c5-72d50e7073b8. PMID: 20186320; PMCID: PMC2826397.
		"""
		# Deviation from DREAM 3:
		# Construct mus and sigmas without the influence of
		# the perturbed observations
		# Including the perturbed experiments seemed to introduce
		# a problematic bias, at least for Genesnake simulated data.
		# It also makes sense from a theoretical standpoint.
		# The perturbations is a different process than
		# the regulation which we are trying to estimate.
		mu_B = []
		sigma_B = []
		for jj in range(P.shape[1]):
			ii_perturbed = P.iloc[:, jj].astype(bool)
			tmp = Y.iloc[:, jj]
			expr_unperturbed = tmp.loc[~ii_perturbed]
			mu_B.append(expr_unperturbed.mean())
			sigma_B.append(expr_unperturbed.std())

		l = []
		# Regulators A indexed by jj
		for jj in range(P.shape[1]):
			ii_perturbed = P.iloc[:, jj].astype(bool)

			# Modification/extension of DREAM 3:
			# Considers that one gene
			# can be perturbed several times. Averages the
			# expression from those perturbations.
			# Average seems like the natural choice for a first attempt.
			expr_perturbed = Y.loc[ii_perturbed, :]
			# Zscores for regulation A -> B, see DREAM 3 paper notation.
			# Note that we are doing all targets B at once for
			# certain regulator A, indexed by jj. Hence whole
			# rows of Y at a time
			zab = (expr_perturbed.mean(axis = 0) - mu_B) / sigma_B
			l.append(zab)
		network = pd.DataFrame(
			data = np.vstack(l),
			index = P.columns,  # Since As indexed by jj over P.columns
			columns = Y.columns  # Since expr_perturbed is from rows from Y
			)
		# Correct the sign. This assumes knockdown data.
		# If A knockdown leads to B overexpression, the regulation
		# is an inhibition, signified by negative sign on the weight.
		# Compare to how overexpression of B gives positive sign above.
		# Might rewrite to consider both knockdown and overexpression data
		# in the future.
		network = -network
		network.index.name = 'regulator'
		network.columns.name = 'target'
		return(network)
	if method == "mean_difference":
		"""

		Mean differences method, from submission to the causalbench challenge;
		CausalBench Challenge: Differences in Mean Expression
		Marcin Kowiel, Wojciech Kotlowski, Dariusz Brzezinski
		https://openreview.net/forum?id=Cx9B85IlEVR

		One of the top performers in the post challenge evaluation paper;
		Chevalley, M., Roohani, Y.H., Mehrjou, A. et al. A large-scale benchmark for network inference from single-cell perturbation data. Commun Biol 8, 412 (2025).
		https://doi.org/10.1038/s42003-025-07764-y

		Implemented myself, since simple, and no code published as far as I've seen.

		This version computes the unperturbed means on all cells where that gene
		is not perturbed. mean_difference_controls_only instead computes it on
		control cells only, ie cells where no gene was perturbed. Judging by the text
		of their report, my best guess is that they used mean_difference_controls_only

		Conventions used for this method:
		Data: [experiments/cells/observations x genes/variables]
		Output network: [regulators x targets]

		Nomenclature; Their original nomenclature used X for regulator and Y
		for target. Since already using Y for expression matrix, (and reusing
		code from the zscore method), below instead uses A for regulator
		and B for target.

		"""
		# Calculate mean values of unperturbed expression for all genes
		mu_B = []
		for jj in range(P.shape[1]):
			ii_perturbed = P.iloc[:, jj].astype(bool)
			tmp = Y.iloc[:, jj]
			expr_unperturbed = tmp.loc[~ii_perturbed]
			mu_B.append(expr_unperturbed.mean())
		mean_differences = []
		# Regulators A indexed by jj
		for jj in range(P.shape[1]):
			ii_perturbed = P.iloc[:, jj].astype(bool)
			expr_perturbed = Y.loc[ii_perturbed, :]
			# Note that we are doing all targets B at once for
			# certain regulator A, indexed by jj. Hence whole
			# rows of Y at a time. Likewise, mu_B is not scalar
			# but the whole array we collected in the previous loop
			mean_difference = expr_perturbed.mean(axis = 0) - mu_B

			mean_differences.append(mean_difference)

		network = pd.DataFrame(
			data = np.vstack(mean_differences),
			index = P.columns,  # Since As indexed by jj over P.columns
			columns = Y.columns  # Since expr_perturbed is from rows from Y
			)
		# Method does unsigned inference
		network = network.abs()
		network.index.name = 'regulator'
		network.columns.name = 'target'
		return(network)
	if method == "mean_difference_controls_only":
		"""

		Mean differences method, from submission to the causalbench challenge;
		CausalBench Challenge: Differences in Mean Expression
		Marcin Kowiel, Wojciech Kotlowski, Dariusz Brzezinski
		https://openreview.net/forum?id=Cx9B85IlEVR

		One of the top performers in the post challenge evaluation paper;
		Chevalley, M., Roohani, Y.H., Mehrjou, A. et al. A large-scale benchmark for network inference from single-cell perturbation data. Commun Biol 8, 412 (2025).
		https://doi.org/10.1038/s42003-025-07764-y

		Implemented myself, since simple, and no code published as far as I've seen.

		Conventions used for this method:
		Data: [experiments/cells/observations x genes/variables]
		Output network: [regulators x targets]

		Nomenclature; Their original nomenclature used X for regulator and Y
		for target. Since already using Y for expression matrix, (and reusing
		code from the zscore method), below instead uses A for regulator
		and B for target.

		"""
		controls_bool = (Y.index == 'control')
		if sum(controls_bool) == 0:
			print('No control samples found in dataset. This method is unsuitable. Returning None instead of network')
			return None
		# Mean values of unperturbed expression for all genes
		mu_B = Y.loc[controls_bool, :].mean(axis = 0)
		mean_differences = []
		# Regulators A indexed by jj
		for jj in range(P.shape[1]):
			ii_perturbed = P.iloc[:, jj].astype(bool)
			expr_perturbed = Y.loc[ii_perturbed, :]
			# Note that we are doing all targets B at once for
			# certain regulator A, indexed by jj. Hence whole
			# rows of Y at a time. Likewise, mu_B is not scalar
			# but the whole array we collected in the previous loop
			mean_difference = expr_perturbed.mean(axis = 0) - mu_B
			mean_differences.append(mean_difference)
		network = pd.DataFrame(
			data = np.vstack(mean_differences),
			index = P.columns,  # Since As indexed by jj over P.columns
			columns = Y.columns  # Since expr_perturbed is from rows from Y
			)
		# Method does unsigned inference
		network = network.abs()
		network.index.name = 'regulator'
		network.columns.name = 'target'
		return(network)
	# I don't trust this:
	# - Why transposing stuff?
	# - Why P2 that is then never used?
	# - Why np.mean(Y, axis=1), ie mean of entire Y, when algorith
	#   uses mean of expressions of gene that is knocked out?
	# Made my own, that I understand, and is better documented,
	# so I'll use that going forward  //AB
	if method == "zscore_old":
		outNet = []
		P2 = P.transpose()
		for i in range(P2.shape[1]):
			inds = np.nonzero(abs(P.iloc[:,i].array))
			Y2 = Y.transpose()
			tmpAest = np.divide(
				(np.mean(Y2.iloc[inds], axis=0) - np.mean(Y, axis=1)), 
				np.std(Y, axis=1))
			outNet.append(tmpAest)
		aest = pd.DataFrame(outNet)
		return aest
	if method == "lasso":  # lasso, works ok
		aests = []
		for z in zvec:
			lasso_mod = Lasso(alpha=z)
			out_net_tmp = []
			for i in range(P.shape[0]):
				lasso_mod.fit(Y.transpose(), -P.iloc[i,:])
				out_net_tmp.append(lasso_mod.coef_)
			out_net_tmp = pd.DataFrame(out_net_tmp)
			aests.append(out_net_tmp)
			return aests
			# Probably broke as shit
			# Note how it returns from this whole function
			# when completing the solution for the first z in zvec
			# The comment above says "works ok". How can that possibly be?
			# It's not even returning according to the intended format..
			# //AB
	if method == 'lasso_density':
		# Returns a lasso network with density close to a target density, 
		# by optimising the alpha parameter using Brents root finding algorithm.
		# Adapted from (the sensical parts of) the lasso method herein
		# by Anton Björk.
		#
		# Format conventions used for this method:
		# Data: [experiments/cells/observations x genes/variables]
		# Output network: [regulators x targets]

		# Transposing for compatibility with lasso code that
		# Uses the opposite conventions.
		P = P.T
		Y = Y.T

		from threadpoolctl import threadpool_limits
		from sklearn.linear_model import Lasso

		def block_format(string_):
			import textwrap
			return(' '.join(textwrap.dedent(string_).splitlines()))
		def perturbation_lasso(Y, P, alpha):
			# Underlying libraries seem to multithread with fast sync cycles,
			# like within single matrix operations, and so care must be taken
			# to not oversubscribe the CPU, because then other computation
			# gets interleaved on the CPU, which grinds the lasso to a halt,
			# because approximately all time is spent waiting for the chunk
			# that got stuck behind the most other stuff. At least that's what
			# I think is going on. Limiting the number of cores scikit expects
			# to be able to access removes the scheduling conflicts /
			# inteleaving of processing, because there are still free cores
			# for the other things to use instead. This makes the (fragile)
			# fast sync cycle multiprocessing run as expected.
			# This is of course an incomplete solution, which depends on
			# how much other stuff is running simultaneously, and how 
			# many cores are available. What one would want to do conceptually
			# is to reserve this number of cores for this thing only,
			# but not sure if that is possible. The OS interleaves by default
			# I think, and I am not sure it first asks the processes if it may.
			with threadpool_limits(limits = 5):
				lasso_mod = Lasso(alpha = alpha)
				out_net_tmp = []
				for i in range(P.shape[0]):
					lasso_mod.fit(Y.transpose(), -P.iloc[i, :])
					out_net_tmp.append(lasso_mod.coef_)
			out_net = pd.DataFrame(
				data = out_net_tmp,
				index = Y.index,
				columns = Y.index)
			return(out_net)
		def get_alpha_for_target_sparsity(
				Y, P,
				target_nonzeros_per_gene,
				):
			def sparsity_deviation(alpha):
			    network = perturbation_lasso(Y, P, alpha)
			    avg_nonzeros = np.mean(np.count_nonzero(network.values, axis=1))
			    deviation = avg_nonzeros - target_nonzeros_per_gene
			    print(f"alpha = {alpha}, avg_nonzeros = {avg_nonzeros}, deviation = {deviation}")
			    return(deviation)
			from scipy.optimize import brentq
			heuristic_lower_bound = 1e-10
			heuristic_upper_bound = 1e2
			try:
				alpha = brentq(
				    sparsity_deviation,
				    a = heuristic_lower_bound,
				    b = heuristic_upper_bound,
				)
			except RuntimeError as e:
				print(e)
				print(
					'Brentq didnt finish with the default maximum iterations. '
					'Trying one more time with higher max iterations'
					)
				alpha = brentq(
				    sparsity_deviation,
				    a = heuristic_lower_bound,
				    b = heuristic_upper_bound,
				    maxiter = 300
				)
			except ValueError as e:
				print(e)
				print(
					'Seems the starting bracket did not include '
					'the optimal sparsity. Trying a wider bracket, '
					'along with higher maximum iterations'
					)
				alpha = brentq(
				    sparsity_deviation,
				    a = 1e-30,
				    b = 1e30,
				    maxiter = 300
				)
			print(f'{alpha = }')
			return(alpha)

		target_density = kwargs['target_density']
		n_genes = Y.shape[0]
		
		if target_density < 0:
			import textwrap
			raise ValueError(textwrap.dedent(
				f'''
				{target_density = }
				Cannot make network with negative density
				'''
				))
		if target_density * n_genes > n_genes ** 2:
			print(block_format(
				'''
				The requested density is higher than a full network.
				Returning full network instead.
				The target density is given as average number of
				links per gene.
				'''
				))
			target_density = n_genes
			network = perturbation_lasso(
				Y = Y,
				P = P,
				alpha = 0
				)
		else:
			alpha = get_alpha_for_target_sparsity(
				Y = Y,
				P = P,
				target_nonzeros_per_gene = target_density,
				)
			network = perturbation_lasso(
				Y = Y,
				P = P,
				alpha = alpha
				)

		avg_nonzeros = np.mean(np.count_nonzero(network.values, axis=1))
		print(f'{avg_nonzeros = }')
		if abs(avg_nonzeros - target_density) > 0.1 * target_density:
			raise ValueError(block_format(
				'''
				The density of the network found differs more than
				10% from the target density.
				'''
				))

		network = network.T
		return(network)
	if method == "elnet":  # elastic net, works semi ok
		aests = []
		for z in zvec:
			lassoMod = ElasticNet(alpha=z, random_state=0)
			outNetTmp = []
			for i in range(P.shape[0]):
				lassoMod.fit(Y.transpose(), -P.iloc[i,:])
				outNetTmp.append(lassoMod.coef_)

			outNetTmp = pd.DataFrame(outNetTmp)
			aests.append(outNetTmp)
			return aests

	if method == "linsvm":  # linear SVM, works ok
		from sklearn.svm import SVR
		aest = []
		for i in range(P.shape[0]):
			svr_lin = SVR(kernel="linear", C=100, gamma="auto")
			fitc = svr_lin.fit(Y.transpose(), -P.iloc[i,:]).coef_
			aest.append(fitc)
		aest = pd.DataFrame(np.concatenate(aest))
		return aest

	if method == "regtrees":  # regression trees, works not ok
		aest = []
		from sklearn.tree import DecisionTreeRegressor

		n = Y.shape[0]
		for i in range(0, n):
			# create a regressor object
			regressor = DecisionTreeRegressor(random_state=0)
			Y2 = Y.transpose()
			fitc = regressor.fit(Y2, Y2.iloc[:, i])
			aest.append(fitc.feature_importances_)
		aest = pd.DataFrame(aest).transpose()
		return aest

	if method == "rbfsvm":  # rbf svm (nonlinear), works not ok
		from sklearn.svm import SVR
		from sklearn.inspection import permutation_importance
		aest = []
		n = Y.shape[0]
		svrr = SVR(kernel="rbf")
		for i in range(0, n):
			Y2 = Y.transpose()
			svrr.fit(Y2, Y2.iloc[:, i])
			perm_importance = permutation_importance(svrr, Y2, Y2.iloc[:, i])
			aest.append(perm_importance['importances_mean'])
		aest = pd.DataFrame(aest)
		return aest

	if method == "polysvm":  # polynomial svm, works not ok
		from sklearn.svm import SVR
		from sklearn.inspection import permutation_importance
		aest = []
		n = Y.shape[0]
		svrp = SVR(kernel="poly")
		for i in range(0, n):
			Y2 = Y.transpose()
			svrp.fit(Y2, Y2.iloc[:, i])
			perm_importance = permutation_importance(svrp, Y2, Y2.iloc[:, i])
			aest.append(perm_importance['importances_mean'])
		aest = pd.DataFrame(aest)
		return aest
	# Put it down here so I don't mistake the other zscore with this
	# when editing. Could become very confusing.
	if method == "zscore_dream3":
		"""
		Most of notation and code according to the zscore method from 
		the DREAM 3 paper. Search for z-score in the paper to find 
		the section.

		Conventions used for this method:
		Data: [experiments/cells/observations x genes/variables]
		Output network: [regulators x targets]

		Returns NaN for genes that are not perturbed in any experiments.

		DREAM 3 paper:
			Link
				https://doi.org/10.1371/journal.pone.0009202
			Citation
				Prill RJ, Marbach D, Saez-Rodriguez J, Sorger PK, Alexopoulos LG, Xue X, Clarke ND, Altan-Bonnet G, Stolovitzky G. Towards a rigorous assessment of systems biology models: the DREAM3 challenges. PLoS One. 2010 Feb 23;5(2):e9202. doi: 10.1371/journal.pone.0009202. Erratum in: PLoS One. 2010;5(3). doi: 10.1371/annotation/f633213a-dc4f-4bee-b6c5-72d50e7073b8. PMID: 20186320; PMCID: PMC2826397.
		"""
		mu_B = Y.mean(axis = 0)
		sigma_B = Y.std(axis = 0)
		# Regulators A indexed by jj
		l = []
		for jj in range(P.shape[1]):
			ii_perturbed = P.iloc[:, jj].astype(bool)
			# Modification/extension of DREAM 3:
			# Considers that one gene
			# can be perturbed several times. Averages the
			# expression from those perturbations.
			# Average seems like the natural choice for a first attempt.
			expr_perturbed = Y.loc[ii_perturbed, :]
			# Zscores for regulation A -> B, see DREAM 3 paper notation.
			# Note that we are doing all targets B at once for
			# certain regulator A, indexed by jj. Hence whole
			# rows of Y at a time
			zab = (expr_perturbed.mean(axis = 0) - mu_B) / sigma_B
			l.append(zab)
		network = pd.DataFrame(
			data = np.vstack(l),
			index = P.columns,  # Since As indexed by jj over P.columns
			columns = Y.columns  # Since expr_perturbed is from rows from Y
			)
		# Correct the sign. This assumes knockdown data.
		# If A knockdown leads to B overexpression, the regulation
		# is an inhibition, signified by negative sign on the weight.
		# Compare to how overexpression of B gives positive sign above.
		# Might rewrite to consider both knockdown and overexpression data
		# in the future.
		network = -network
		network.index.name = 'regulator'
		network.columns.name = 'target'
		return(network)	

	raise ValueError('Unknown method name')





#
# def compare_networks(nets, net0, exclude_diagonal=False, signed=False):
#     """
#     Function for comparing networks and returning performance statistics.
#     Allows to exclude diagonal and/or consider sign in a network.
#     INPUT:
#         nets - inferred networks
#         net0 - gold standard network for comparison
#     """
#     TPo = []
#     FPo = []
#     TNo = []
#     FNo = []
#
#     TPRo = []
#     TNRo = []
#     PPVo = []
#     NPVo = []
#     FNRo = []
#     FPRo = []
#     FDRo = []
#     FORo = []
#     LRpo = []
#     LRmo = []
#     PTo = []
#     TSo = []
#
#     F1o = []
#     MCCo = []
#     FMo = []
#     BMo = []
#     MKo = []
#     DORo = []
#     So = []
#
#     for n in range(0, len(nets)):
#
#         if (exclude_diagonal):  # if diagonal should be excluded for statistics
#             net1 = np.array(nets[n])  # take a single network to compare
#             np.fill_diagonal(net1, 0)
#             net2 = np.array(net0)  # gold standard
#             np.fill_diagonal(net2, 0)
#         else:  # if diagonal should be included for statistics
#             net1 = np.array(nets[n])  # take a single network to compare
#             net2 = np.array(net0)  # gold standard
#
#         S = len(np.nonzero(net1)[0])/net1.shape[0] # sparsity of a network
#
#         So.append(S)
#
#         if (signed):  # if sign should be considered
#             net1[net1 > 0] = 1
#             net1[net1 < 0] = -1
#             net2[net2 > 0] = 1
#             net2[net2 < 0] = -1
#             cond1 = net1 == net2
#             cond2 = net1 != 0
#             cond3 = net2 != 0
#
#             TP = sum(sum(cond1 & cond2 & cond3))
#             TPo.append(TP)
#             FP = sum(sum(net1.astype(dtype=bool))) - TP
#             FPo.append(FP)
#             TN = sum(sum(~net1.astype(dtype=bool) & ~net1.astype(dtype=bool)))
#             TNo.append(TN)
#             FN = sum(sum(~net1.astype(dtype=bool))) - TN
#             FNo.append(FN)
#         else:  # if sign should be ignored
#             net1 = net1.astype(dtype=bool)
#             net2 = net2.astype(dtype=bool)
#             TP = sum(sum(net1 & net2))
#             TPo.append(TP)
#             FP = sum(sum(net1)) - TP
#             FPo.append(FP)
#             TN = sum(sum(~net1 & ~net2))
#             TNo.append(TN)
#             FN = sum(sum(~net1)) - TN
#             FNo.append(FN)
#
#         #precision, recall, thresholds = precision_recall_curve((net2+[0]).flatten(), (net1+[0]).flatten())
#         # Use AUC function to calculate the area under the curve of precision recall curve
#         #auc_precision_recall = auc(recall, precision)
#         #roc_auc = roc_auc_score((net2+[0]).flatten(), (net1+[0]).flatten())
#
#         # AUPRo.append(auc_precision_recall)
#         # AUCo.append(roc_auc)
#
#         # measures based on confusion matrix
#         # based on https://en.wikipedia.org/wiki/Confusion_matrix
#         # ifs are for handling division by zero
#
#         # sensitivity
#         if (TP + FN) != 0:
#             TPR = TP / (TP + FN)
#         else:
#             TPR = 0
#         TPRo.append(TPR)
#         # specificity
#         if (TN + FP) != 0:
#             TNR = TN / (TN + FP)
#         else:
#             TNR = 0
#         TNRo.append(TNR)
#         # precision
#         if (TP + FP) != 0:
#             PPV = TP / (TP + FP)
#         else:
#             PPV = 0
#         PPVo.append(PPV)
#         # negative predictive value
#         if (TN + FN) != 0:
#             NPV = TN / (TN + FN)
#         else:
#             NPV = 0
#         NPVo.append(NPV)
#         # miss rate
#         if (FN + TP) != 0:
#             FNR = FN / (FN + TP)
#         else:
#             FNR = 0
#         FNRo.append(FNR)
#         # fall-out
#         if (FP + TN) != 0:
#             FPR = FP / (FP + TN)
#         else:
#             FPR = 0
#         FPRo.append(FPR)
#         # false discovery rate
#         if (FP + TP) != 0:
#             FDR = FP / (FP + TP)
#         else:
#             FPR = 0
#         FDRo.append(FDR)
#         # false omission rate
#         if (FN + TN) != 0:
#             FOR = FN / (FN + TN)
#         else:
#             FOR = 0
#         FORo.append(FOR)
#         # positive likelihood ratio
#         if FPR != 0:
#             LRp = TPR / FPR
#         else:
#             LRp = 0
#         LRpo.append(LRp)
#         # negative likelihood ratio
#         if TNR != 0:
#             LRm = FNR / TNR
#         else:
#             LRm = 0
#         LRmo.append(LRm)
#         # prevalence threshold
#         if (np.sqrt(TPR) + np.sqrt(FPR)) != 0:
#             PT = np.sqrt(FPR) / (np.sqrt(TPR) + np.sqrt(FPR))
#         else:
#             PT = 0
#         PTo.append(PT)
#         # threat score
#         if (TP + FN + FP) != 0:
#             TS = TP / (TP + FN + FP)
#         else:
#             TS = 0
#         TSo.append(TS)
#         # F1 score
#         if (2 * TP + FP + FN) != 0:
#             F1 = 2 * TP / (2 * TP + FP + FN)
#         else:
#             F1 = 0
#         F1o.append(F1)
#         # MCC
#         MCC = np.sqrt(PPV * TPR * TNR * NPV) - np.sqrt(FDR * FNR * FPR * FOR)
#         MCCo.append(MCC)
#         # Fowlkes-Mallows index
#         FM = np.sqrt(PPV * TPR)
#         FMo.append(FM)
#         # informedness
#         BM = TPR + TNR - 1
#         BMo.append(BM)
#         # markedness
#         MK = PPV + NPV - 1
#         MKo.append(MK)
#         #diagnostic odds ratio
#         if LRm != 0:
#             DOR = LRp / LRm
#             DORo.append(DOR)
#         else:
#             DORo.append(0)
#
#     # calculating AUPR and AUROC
#
#     srt = np.argsort(np.transpose(np.asarray(So)))
#     TPRate = np.transpose(np.asarray(TPRo))
#     FPRate = np.transpose(np.asarray(FPRo))
#     Prec = np.transpose(np.asarray(PPVo))
#
#     AUROCo = round(np.trapz(TPRate[srt], FPRate[srt]), ndigits=4)
#     AUPRo = round(np.trapz(Prec[srt], TPRate[srt]), ndigits=4)
#
#     data = [TPo, FPo, TNo, FNo,
#         TPRo, TNRo, PPVo, NPVo, FNRo, FPRo, FDRo, FORo, LRpo, LRmo, PTo, TSo,
#         F1o, MCCo, FMo, BMo, MKo, DORo, So]
#     df = pd.DataFrame(data,
#                     index=['tp', 'fp', 'tn', 'fn',
#                          'tpr', 'tnr', 'ppv', 'npv', 'fnr', 'fpr', 'fdr', 'for', 'lrp', 'lrm','pt','ts',
#                          'f1', 'mcc', 'fm', 'bm', 'mk', 'dor', 's'])
#     op = dict()
#     df2 = df.replace(np.nan, 0)
#     op['metrics'] = df2.transpose()
#     op['AUROC'] = AUROCo
#     op['AUPR'] = AUPRo
#     return op


# def data_summary(data, p_matrix):
#     """
#     Function for displaying data summary.
#     INPUT:
#         data - data matrix (Y)
#         p_matrix - perturbation matrix (P)
#     """
#
#     nrw = len(data.index)
#     ncl = len(data.columns)
#
#     data2 = data.to_numpy()
#     test_nor = kstest(data2.flatten(), 'norm')
#
#     desc = list(['0 values', 'Inf values', 'NaN values', 'positive values', 'negative values'])
#     vals = np.asarray([np.isclose(data, 0.0).sum(),
#            np.isinf(data).sum().sum(),
#            data.isnull().values.sum(),
#            (data > 0).values.sum(),
#            (data < 0).values.sum()])/(nrw*ncl)*100
#
#     # perturbation ranking
#     #for i in range(0, p_matrix.shape[1]-1):
#     #    print(np.array(p_matrix.ix[:,i]))
#
#     pert_rank = []
#
#     for i, column in enumerate(p_matrix):
#         ind_p = p_matrix.iloc[:,i] == 1
#         tmp_data_col = data[i]
#         pert = abs(np.array(tmp_data_col[np.invert(ind_p)])) > abs(np.array(tmp_data_col[ind_p]))
#         #print(sum(pert == True))
#         #print(type(pert))
#         tmp_sum = sum(pert == True).tolist()
#         #print((tmp_sum[0]))
#         pert_rank.append(tmp_sum)
#
#
#     # For Sine Function
#
#     figure, axes = plt.subplots(2, 2, figsize=(12, 12))
#     sns.kdeplot(np.array(data).flatten(), ax = axes[0, 0])
#     axes[0, 0].title.set_text('Data density')
#     sm.qqplot(data, line='45', ax = axes[0, 1])
#     axes[0, 1].text(-1.5, 1.5, 'KS P value = '+str(round(test_nor.pvalue, ndigits=2)), fontsize=12)
#     axes[0, 1].title.set_text('Data normality')
#     axes[1, 0].bar(desc, vals, color ='slategrey', width = 0.4)
#     axes[1, 0].title.set_text('Values in the data')
#     axes[1, 0].set_ylabel("Amount [%]")
#     axes[1, 0].set_xticklabels(desc, rotation=45)
#     # Combine all the operations and display
#     sns.histplot(pert_rank, ax = axes[1, 1], kde=True, element="step")
#     axes[1, 1].title.set_text('Perturbation ranking')
#     plt.show()
#
# import random
# import pandas as pd
# import numpy as np
# from pathlib import Path
# from numpy import inf
# import statistics as stats
# def generate_sequencing_noise(data, SNR, genes_ind="random",  c=5, L=75, techrep=2, ermin=0.45, ermax=0.73, prob_zi=0.1, prob_nb=0.5, sf=0.05):
#     """
#     Function for generating sequencing-based noise based on coverage, GC-content and sequencing error rates
#     INPUT:
#         ng - number of genes
#         nsamp - number of samples
#         genes_ind - indices of genes
#         c - coverage
#         L - read length [bp]
#         techrep - number of technical replicates
#         ermin - minimum sequencing error rate, 0-1
#         ermax - maximum sequencing error rate, 0-1
#         prob_zi - probability of zero-inflation, 0-1
#         prob_nb - probability of noiseless data from negative binomial distribution, e.g. value around 1 is lack of noise
#         sf - scaling factor that adjusts variance in data to variance in noise
#     """
#     nsamp = np.shape(data)[1]
#     ng = np.shape(data)[0]
#     if SNR == "very_low":
#         ermin = 0.7
#         ermax = 1
#         prob_zi = 0.4
#         prob_nb = 0.1
#         sf = 0.005
#     elif SNR == "low":
#         ermin = 0.6
#         ermax = 0.9
#         prob_zi = 0.3
#         prob_nb = 0.1
#         sf = 0.01
#     elif SNR == "medium":
#         ermin = 0.45
#         ermax = 0.73
#         prob_zi = 0.1
#         prob_nb = 0.5
#         sf = 0.05
#     elif SNR == "high":
#         ermin = 0.1
#         ermax = 0.2
#         prob_zi = 0.1
#         prob_nb = 0.5
#         sf = 0.1
#     elif SNR == "very_high":
#         ermin = 0
#         ermax = 0.1
#         prob_zi = 0.05
#         prob_nb = 0.7
#         sf = 0.5
#     else:
#         print("Default noise settings have been used (Wang et al., 2012)")
#     # based on: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-13-185
#     # e.g. here for ENCODE project mismatch counting is 0.45-073
#
#     # code block 3
#     p = Path(__file__).with_name("pc_genes_start_end_gc.txt")
#     df = pd.read_csv(p, sep='\t', header=0)
#
#     # genes lengths
#     g_length = df["end_position"]-df["start_position"]
#     # genes gc content
#     gc_cont = df["percentage_gene_gc_content"]
#
#     #number of genes
#     #ng = 10  # number of genes
#     #c = 1  # RNA-seq coverage
#     #L = 75  # read length usually 50-75 bp for gene expression / RNA profiling
#     #techrep = 2
#     # randomize genes
#     rand_genes_ind = genes_ind
#     if genes_ind == "random":
#         rand_genes_ind = np.random.randint(1, len(g_length), size=ng)
#     else:
#         rand_genes_ind = genes_ind
#     #rand_genes_ind = [1000, 2000, 10000, 10010, 5000, 7000, 7500, 100, 3500, 9000]
#
#     g_start = df["start_position"][rand_genes_ind].to_frame()
#     g_end = df["end_position"][rand_genes_ind].to_frame()
#     # total length of the sequenced genome or genome fragment
#     # in this code, it is a sum of all N genes lengths
#     G = sum(g_length[rand_genes_ind])
#
#     # now we use Lander-Waterman equation to estimate number of reads N for a given coverage
#     N = round((c*G)/L)
#
#     # randomize read positions from unif distribution
#     sr = pd.Series(1-gc_cont[rand_genes_ind]/100)
#     prbs = sr.repeat(g_length[rand_genes_ind])
#     prbs /= prbs.sum()  # need to normalize it so it sums to 1
#     frag = np.arange(0, G) # fragment of sequenced genome
#     g_length_set = g_length[rand_genes_ind] # subset of genes lengths
#
#     #nsamp = 10
#
#     for t in range(0, techrep):
#         allCounts = []
#         allNoErrorCounts = []
#         for s in range(0, nsamp):
#             # error rate of reads
#             # based on: https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-13-185
#             # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4631051/
#             er = random.uniform(ermin, ermax)  # good mapping rates 0.1-0.2 (good) or 0-0.1 (very good)
#             # e.g. here for ENCODE project mismatch counting is 0.45-073
#             # https: // bmcbioinformatics.biomedcentral.com / articles / 10.1186 / 1471 - 2105 - 13 - 185 / tables / 3
#             Ne = round(N - N * er)  # we decrease the number of reads as some of them are mismatched with a given error rate
#             reads_pos = np.random.choice(frag, p=prbs, size=Ne, replace=True)
#             counts_samp = []
#             allCounts.append([])
#             allNoErrorCounts.append([])
#             tmp_start = 0
#             for n in range(0, len(g_length_set)):  # loop over genes
#                 tmp_len = g_length_set.iloc[n]  # tmp variables are to slide over all genes and check how many reads are mapped
#                 tmp_end = tmp_start + tmp_len
#                 tmp_counts = (reads_pos > int(tmp_start + L)) & (reads_pos < int(tmp_end - L))
#                 tmp_start = tmp_end
#                 counts_samp.append(sum(tmp_counts))
#             counts = pd.Series(counts_samp)
#             necounts = pd.Series(c * g_length_set / L)
#
#             allCounts[s].append(np.transpose(counts))
#             allNoErrorCounts[s].append(np.transpose(necounts))
#         out_counts = np.transpose(pd.DataFrame(np.concatenate(allCounts)))  # samples in columns, genes in rows
#         out_necounts = np.transpose(pd.DataFrame(np.concatenate(allNoErrorCounts)))  # samples in columns, genes in rows
#     out_counts += out_counts
#     out_necounts += out_necounts
#
#     # zero inflation negative binomial noise to genes
#     # zero inflation simulates dropouts
#     # negative binomial simulates noise strength in counts
#
#     def zinbme(data, prob_nb, sf): # sf is scaling factor (SNR like)
#         n = round(mean(data)-st.stdev(data))
#         nb = np.random.negative_binomial(n, prob_nb, size=len(data))  # negative binomial distribution
#         zinb = nb
#         msf = st.sqrt(np.var(data) / (sf * (np.var(zinb))))  # scaling factor to adjust noise values to data
#         return pd.Series(zinb*msf)
#
#     out_counts = pd.DataFrame(round(out_counts/techrep))
#     out_error = out_counts.apply(lambda x: zinbme(x, prob_nb, sf), axis=1)
#
#     ratios = np.divide(out_counts+out_error, out_necounts / techrep)
#     ratios = ratios.apply(lambda x: np.multiply(x, np.random.choice([0, 1], p=[prob_zi, 1 - prob_zi], size=len(x), replace=True)), axis=1)
#     ratios[ratios == inf] = 0
#     out_counts[out_counts == inf] = 0
#     op = dict()
#     op['ratios'] = ratios
#     op['seq_counts'] = round(out_counts)
#     op['noerror_counts'] = round(out_necounts / techrep)
#     op['zinb_counts'] = out_error
#
#     return op
