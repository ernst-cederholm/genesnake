
def calculate_f1_scores(
		estimated_network,
		reference_network,
		):

	import numpy as np
	from sklearn.metrics import precision_recall_curve

	precision, recall, thresholds = precision_recall_curve(
		y_true = reference_network, 
		y_score = estimated_network)

	# Reverse them so that the "start point" of the precision recall
	# curve is at the start of the arrays.
	# I consider low recall values the start point. Not needed for correctness,
	# but easier to look at and reason about the data if the layout
	# matches my conceptual view of it
	#
	# Precision recall curve adds a starting point (0, 1).
	# This is an assumption or convention and not derived from the data.
	# I'd rather start with first (recall, precision) point defined by the data.
	#
	# Both precision and recall can be 0. The f1 for that point is not
	# defined, division by 0. Hence remove that point if it exists.
	#
	# Removing the false start point before flipping in a misguided
	# premature optimisation. I think appends and removes
	# from ends of arrays have O(1) complexity while those at beginning
	# have O(n). It most likely doesn't matter. Sorry, can't help it.
	precision = precision[:-1]
	recall = recall[:-1]
	ii_defined_f1 = np.logical_not((precision == 0) & (recall == 0))
	precision = precision[ii_defined_f1]
	recall = recall[ii_defined_f1]
	precision = np.flip(precision)
	recall = np.flip(recall)

	f1_scores = 2 * (precision * recall) / (precision + recall)
	max_f1 = np.max(f1_scores)

	return(
		{
			'f1_scores': f1_scores,
			'max_f1': max_f1
			}
		)
