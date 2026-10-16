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





