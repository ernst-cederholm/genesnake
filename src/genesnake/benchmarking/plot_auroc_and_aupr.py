
import matplotlib.pyplot as plt
from pathlib import Path

def plot_auroc_and_aupr(
		stats,
		model_name,
		output_dir,
		fix_ylim = False
		):

	plot_metrics = stats['plot_metrics']
	Path(output_dir).mkdir(exist_ok = True)

	plt.close()
	plt.plot(
		plot_metrics['FPRate'], plot_metrics['TPRate'], 
		color = 'tab:blue', linestyle = 'solid',
		label = 'Observed')
	plt.plot(
		plot_metrics['FPRate_extension'], 
		plot_metrics['TPRate_extension'], 
		color = 'tab:blue', linestyle = 'dashed',
		label = 'Extended'
		)
	plt.xlim((0, 1))
	plt.ylim((0, 1.01))
	plt.xlabel('FP Rate')
	plt.ylabel('TP Rate')
	plt.legend()
	plt.text(
		0.4, 
		0.1, 
		f'AUROC (extended) = {stats["AUROC"]:.3}',
		)
	title = f'{model_name}_ROC_curve'
	plt.title(title)
	plt.tight_layout()
	filename = f'{title}.svg'
	plt.savefig(output_dir / filename)

	plt.close()
	plt.plot(plot_metrics['Recall'], plot_metrics['Precision'],
		color = 'tab:blue', linestyle = 'solid',
		label = 'Observed')
	plt.plot(
		plot_metrics['Recall_extension'], 
		plot_metrics['Precision_extension'],
		color = 'tab:blue', linestyle = 'dashed',
		label = 'Extended')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.xlim((0, 1))
	# When AUPR is really shit, one cannot see the shape of it
	# because it's just a line at the bottom. But the shape is 
	# useful to check if shape looks sane. Thus the option
	if fix_ylim:
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
	title = f'{model_name}_PR_curve'
	plt.title(title)
	plt.tight_layout()
	filename = f'{title}.svg'
	plt.savefig(
		output_dir / filename)

