
import numpy as np
import genesnake as gs
import copy


def early_precision_rate(
		inferred_network, 
		reference_network,
		fraction_of_P = 1):
	"""
	Calculate Early Precision Rate (EPR) according to
	dspin bioarxiv preprint at
	doi
		https://doi.org/10.1101/2023.04.19.537364
	link (more precise, to full text version 4)
		https://www.biorxiv.org/content/10.1101/2023.04.19.537364v4.full

	Given reading a few lines of the dspin bioarxiv, I think the code
	with comments below will be understandable, so leaving it at that.
	Get in touch if problems though! You not understanding means I 
	haven't documented well enough.  //Anton Björk
	"""

	top_k_accuracy_ = top_k_accuracy(
		inferred_network, 
		reference_network,
		fraction_of_P = fraction_of_P)
	# Using ERMA to calculate the precision of a random model since
	# they are the same. See calculate_ERMA documentation for explanation
	# if curious.
	ERMA = gs.benchmarking.calculate_ERMA(reference_network)
	if ERMA == 0:
		raise ValueError(
                'ERMA is 0, which implies that the reference network has density 0. Something has gone wrong')
	else:
		early_precision_rate = top_k_accuracy_ / ERMA
	return(early_precision_rate)


def top_k_accuracy(
		inferred_network, 
		reference_network,
		fraction_of_P = 1):
	"""
	Calculate Top K accuracy according to
	dspin bioarxiv preprint at
	doi
		https://doi.org/10.1101/2023.04.19.537364
	link (more precise, to full text version 4)
		https://www.biorxiv.org/content/10.1101/2023.04.19.537364v4.full

	Given reading a few lines of the dspin bioarxiv, I think this code
	will be understandable, so leaving it at that.
	Get in touch if problems though! You not understanding means I 
	haven't documented well enough.  //Anton Björk

	Update:

	There are a few more things worth understanding about this one.
	Nomenclature:
		P = positive, ie edge exists
		N = negative, ie edge does not exist
		prefix P: predicted
		prefix T: True, ie correct, in the context of predictions
		prefix F: False, ie wrong, in the context of predictions
	accuracy = (TP + TN) / (P + N).
	When people talk about top k accuracy (or top K metrics in general),
	they substitute P + N for PP + PN = K. Otherwise, no model could have
	a good score when K << P + N, as it typically is, since otherwise no point with
	top K metrics. Thus
	top K accuracy = (TP + TN) / (PP + PN) = (TP + TN) / K.
	When originally implemented, we considered the case of predictions
	with PP > K. In this case, accuracy reduces to precision, since
	the top K list is fully filled with PP, so PN and thus TN are impossible, so
	top K accuracy = (TP + 0) / (PP + 0) = TP / PP = precision.
	Because of this, it's implemented as precision in this function.
	So far, so good.

	Recently we ran some very sparse lassos, and encountered PP < K.
	Here, there is a problem. If one goes with standard accuracy,
	top K accuracy = (TP + TN) / (PP + PN) = (TP + TN) / K.
	When PP < K, K - PP predictions are negative, ie edge does not exist.
	The first problem then is how to rank those when they are all 0,
	because what is asked for is a list of top K predictions. Here one has to
	choose how to handle the situation. One perspective is that there is
	an inconsistency in demanding the top K PP, when PP < K, and so refuse.
	Otherwise, the natural choice would be to pick randomly.
	Because GRN inference is a sparse problem, most
	edges are N, so almost all of the K - PP PN edges will be scored TN. Thus,
	one encounters the common reason that people don't use accuracy as metric
	for sparse problems; Accuracy does not take class imbalance into account,
	and thus it's for example trivial to create a high scoring method,
	by simply making all predictions the majority class N, or in other words,
	predict an empty network. Thus, it's not an informative metric.
	Because of this, I think that the people who inspired us to use top k accuracy
	also didn't consider the case PP < K, and never encountered this problem.

	I chose to handle the problem by leaving this metric untouched,
	while introducing another metric, see precision_at_k. Thus, this function
	calculates the precision rather than accuracy of the (at most) top K
	edges. A consequence is that a method can achieve a score as high as it's
	global precision, even though it ouputs PP < K.
	"""
	K = round(len(np.array(reference_network).nonzero()[0]) * fraction_of_P)
	if K == 0:
		import warnings
		warnings.warn('There are no edges in the reference network')
	abs_net = np.abs(np.array(inferred_network))
	cutoff = sorted(abs_net.flatten(), reverse = True)[K - 1]
	top_K_network = copy.deepcopy(inferred_network)
	top_K_network[abs_net < cutoff] = 0

	net1 = top_K_network.astype(dtype=bool)
	# Should already be bool, but eh
	net2 = reference_network.astype(dtype=bool)

	TP = (net1 & net2).sum()
	FP = net1.sum() - TP
	TN = (~net1 & ~net2).sum()
	FN = (~net1).sum() - TN

	# precision
	if (TP + FP) == 0:
		precision = np.nan
	else:
		precision = TP / (TP + FP)

	return(float(precision))



def precision_at_k(
		inferred_network,
		reference_network,
		fraction_of_P = 1):
	"""
	For background, see the docstring of top_k_accuracy.

	Another choice for handling the problem when PP < K is to choose to not
	score PN in the top K list, and thus setting TN = 0. Then,
	metric = (TP + 0) / K = TP / K.
	The bad part is that this metric is neither accuracy nor precision.
	A good part is that it handles the problem of TN in such
	a way that one avoids the issues with using accuracy for class imbalanced
	problems. One can not trivially game this metric for sparse problems, and
	so it is infomative in that sense.
	This metric seems to be commonly known as 'precision at K',
	so I adopted that name to be as consistent and conventional as possible.

	For PP > K, this metric equals the precision of the top K predictions.
	This is probably where it gets it's name from, and I am guessing that
	people who use it also rarely encounter PP < K. For PP > K, it's also
	equal to top_k_accuracy.

	For PP < K, this metric will be < precision. The difference will increase
	with K - PP. One way to think about it is that the metric penalises
	models that are too sparse, by substituting a score 0 for the K - PP
	predictions they didn't make.
	Thus, sparse models that achieves high precision in exactly
	the way they are designed to, by being sparse, will not score well
	in this metric, despite the metric having precision in the name, a metric
	that is commonly used and suitable for sparse problems and predictions.
	This risks masking interesting performances by sparse
	models that one would like to see and know about.

	Being a "top K" metric, the intuition is that it's about the "early"
	performance of models, a small subset of the most confident predictions.
	Depending on how K is set, that can probably be the case. As we've
	seen it used in GRN inference though, K is usually set to P, the number of
	positive / existing edges in the reference network.
	That choice of K means that a model needs both recall 1 and precision 1 to
	reach 1 in precision at K.
	In fact, precision = recall, because for top K metrics, and with K = P,
	precision = TP / PP = TP / K = TP / P = recall
	Fixing precision = recall, and thus requiring good recall for good score
	does not match the intuition about the metric having to do with "early"
	performance. One perspective might be that the notion of "early" performance
	is inherently tied to the precision of the most confident predictions,
	and this metric is not precision. Of course, the blame for this might
	be on the selection of K rather than the metric itself. For K << P,
	high precision at K does not require high recall.

	Since precision at K with K = P is as much about recall
	as precision (exactly actually), it seems similar to other well known metrics,
	like AUPR and f1 scores. The practical tests I did also showed that
	precision at K behaves similar to AUPR. This makes is less useful in
	practice, conveying similar information.

	So, the paragraphs above explain why I left the original
	top K accuracy untouched (despite its admittedly poor name), and
	instead introduced this metric as an option alongside it.
	"""
	reference_network = reference_network.astype(dtype=bool)
	K = sum(reference_network)
	K = round(K * fraction_of_P)

	if K == 0:
		import warnings
		warnings.warn('There are no edges in the reference network')
		p_at_k = np.nan
		return p_at_k

	tmp = inferred_network
	iis = sorted(range(len(tmp)), key = lambda i: tmp[i], reverse = True)
	ref = reference_network[iis][:K]
	inf = inferred_network[iis][:K].astype(dtype = bool)

	TP = sum(inf & ref)
	p_at_k = TP / K

	return p_at_k


