
import matplotlib.pyplot as plt

def plot_auroc_and_aupr(
		stats,
		method_name = None,
		fix_pr_ylim = False
		):

	def make_title(method_name, suffix):
		if method_name is not None:
			title = f'{method_name}_{suffix}'
		else:
			title = suffix
		return(title)

	plot_metrics = stats['plot_metrics']
	
	# Create ROC plot
	plot_roc = plt.figure()
	plt.plot(
		plot_metrics['fpr'], plot_metrics['tpr'],
		color = 'tab:blue', linestyle = 'solid',
		label = 'Observed')
	plt.plot(
		plot_metrics['fpr_extension'],
		plot_metrics['tpr_extension'],
		color = 'tab:blue', linestyle = 'dashed',
		label = 'Extended'
		)
	plt.xlim((0, 1))
	plt.ylim((0, 1.01))
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.legend()
	plt.text(
		0.4, 
		0.1, 
		f'AUROC (extended) = {stats["AUROC"]:.3}',
		)
	title = make_title(method_name, 'ROC')
	plt.title(title)
	plt.tight_layout()





	# Create PR plot
	plot_pr = plt.figure()
	plt.plot(plot_metrics['recall'], plot_metrics['precision'],
		color = 'tab:blue', linestyle = 'solid',
		label = 'Observed')
	plt.plot(
		plot_metrics['recall_extension'],
		plot_metrics['precision_extension'],
		color = 'tab:blue', linestyle = 'dashed',
		label = 'Extended')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.xlim((0, 1))
	# When AUPR is really shit, one cannot see the shape of it
	# because it's just a line at the bottom. But the shape is 
	# useful to check if shape looks sane. Thus the option
	if fix_pr_ylim:
		top = 1.1
		plt.ylim((0, top))
	else:
		ax = plt.gca()
		_, top = ax.get_ylim()
		ax.set_ylim(0, top * 1.1)
	plt.legend()
	l1 = f'AUPR (DREAM2 extended) = {stats["AUPR"]:.3}'
	l2 = f'ERMA = {stats["ERMA"]:.3}'
	plt.text(
		0.4, top * 0.7,
		'\n'.join([l1, l2]),
		)
	title = make_title(method_name, 'PR')
	plt.title(title)
	plt.tight_layout()
	
	return plot_roc, plot_pr
