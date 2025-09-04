
def calculate_f1_scores(
		estimated_network,
		reference_network,
		):

	import numpy as np
	from sklearn.metrics import precision_recall_curve

	precision, recall, thresholds = precision_recall_curve(
		y_true = reference_network, 
		y_score = estimated_network)
	f1_scores = 2 * (precision * recall) / (precision + recall)
	best_threshold = thresholds[np.argmax(f1_scores)]
	max_f1 = np.max(f1_scores)

	return(
		{
			'f1_scores': f1_scores,
			'max_f1': max_f1
			}
		)
