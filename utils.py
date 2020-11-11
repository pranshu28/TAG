import uuid
import argparse
import matplotlib

matplotlib.use('Agg')
from models import *
from data_utils import *

import seaborn as sns
import matplotlib.pyplot as plt


TRIAL_ID = uuid.uuid4().hex.upper()[0:6]
# EXPERIMENT_DIRECTORY = './outputs/{}'.format(TRIAL_ID)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def apply_mask(mem_y, out, n_classes):
	for i,y in enumerate(mem_y):
		mask = torch.zeros_like(out[i])
		mask[y-(y%n_classes):y-(y%n_classes)+n_classes]=1
		out[i] = out[i].masked_fill(mask == 0, -1e10)
	return out



def parse_arguments():
	parser = argparse.ArgumentParser(description='Argument parser')
	parser.add_argument('--tasks', default=5, type=int, help='total number of tasks')
	parser.add_argument('--epochs-per-task', default=1, type=int, help='epochs per task')
	parser.add_argument('--dataset', default='rot-mnist', type=str, help='dataset. options: rot-mnist, perm-mnist, cifar100')
	parser.add_argument('--batch-size', default=10, type=int, help='batch-size')
	parser.add_argument('--opt', default=None, type=str, help='Manual adagrad')
	parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
	parser.add_argument('--gamma', default=0.4, type=float, help='learning rate decay. Use 1.0 for no decay')
	parser.add_argument('--dropout', default=0.25, type=float, help='dropout probability. Use 0.0 for no dropout')
	parser.add_argument('--hiddens', default=256, type=int, help='num of hidden neurons in each layer of a 2-layer MLP')
	parser.add_argument('--compute-eigenspectrum', default=False, type=bool, help='compute eigenvalues/eigenvectors?')
	parser.add_argument('--b', default=None, type=int, help='b')
	parser.add_argument('--seed', default=1234, type=int, help='random seed')
	parser.add_argument('--runs', default=3, type=int, help='# runs')
	parser.add_argument('--tag-opt', default='rms', type=str, help='man opt')
	parser.add_argument('--mem-size', default=1, type=int, help='mem')
	parser.add_argument('--multi', default=0, type=int, help='MTL')

	args = parser.parse_args()
	print("Parameters:\n  benchmark="+str(args.dataset)+"\n  num_tasks="+str(args.tasks)+"\n  "+
		  "runs="+str(args.runs)+"\n  batch_size="+str(args.batch_size)+"\n  "+
		  "learning_rate="+str(args.lr)+"\n  learning rate decay(gamma)="+str(args.gamma)+"\n  dropout prob="+str(args.dropout)+"\n  optimizer opt="+str(args.opt))
	if args.opt=='er' or args.opt=='agem':
		print("  mem="+str(args.mem_size))
	elif args.opt=='param':
		print("  tag-opt="+str(args.man_opt))
		print("  b="+str(args.b))
	return args


def init_experiment(args):
	print('\n------------------- Experiment-'+str(args.seed)+' started -----------------')
	# 1. setup seed for reproducibility
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	# 2. create directory to save results
	# Path(EXPERIMENT_DIRECTORY).mkdir(parents=True, exist_ok=True)
	# print("The results will be saved in {}\n".format(EXPERIMENT_DIRECTORY))
	
	# 3. create data structures to store metrics
	loss_db = {t: [0 for i in range(args.tasks*args.epochs_per_task)] for t in range(1, args.tasks+1)}
	acc_db =  {t: [0 for i in range(args.tasks*args.epochs_per_task)] for t in range(1, args.tasks+1)}
	hessian_eig_db = {}
	return acc_db, loss_db, hessian_eig_db


def end_experiment(args, acc_db, loss_db, hessian_eig_db):
	
	# 1. save all metrics into csv file
	# acc_df = pd.DataFrame(acc_db)
	# acc_df.to_csv(EXPERIMENT_DIRECTORY+'/accs.csv')
	# visualize_result(acc_df, EXPERIMENT_DIRECTORY+'/accs.png')
	#
	# loss_df = pd.DataFrame(loss_db)
	# loss_df.to_csv(EXPERIMENT_DIRECTORY+'/loss.csv')
	# visualize_result(loss_df, EXPERIMENT_DIRECTORY+'/loss.png')
	#
	# hessian_df = pd.DataFrame(hessian_eig_db)
	# hessian_df.to_csv(EXPERIMENT_DIRECTORY+'/hessian_eigs.csv')
	
	# 2. calculate average accuracy and forgetting (c.f. ``evaluation`` section in our paper)
	score = np.mean([acc_db[i][-1] for i in acc_db.keys()])
	forget = np.mean([max(acc_db[i])-acc_db[i][-1] for i in range(1, args.tasks)])/100.0
	
	print('average accuracy = {}, forget = {}'.format(score, forget))
	return score, forget


def get_benchmark_data_loader(args):
	"""
    Returns the benchmark loader which could be either of these:
    get_split_cifar100_tasks, get_permuted_mnist_tasks, or get_rotated_mnist_tasks

    :param args:
    :return: a function which when called, returns all tasks
    """
	if args.dataset == 'perm-mnist' or args.dataset == 'permuted-mnist':
		return get_permuted_mnist_tasks
	elif args.dataset == 'rot-mnist' or args.dataset == 'rotation-mnist':
		return get_rotated_mnist_tasks
	elif args.dataset == 'cifar-100' or args.dataset == 'cifar100':
		return get_split_cifar100_tasks
	elif args.dataset == 'imagenet' or args.dataset == 'mini-imagenet':
		return get_mini_imagenet_tasks
	elif args.dataset == 'cub':
		return get_split_cub_tasks
	else:
		raise Exception("Unknown dataset.\n" + "The code supports 'perm-mnist, rot-mnist, and cifar-100.")


def get_benchmark_model(args):
	"""
    Return the corresponding PyTorch model for experiment
    :param args:
    :return:
    """
	if 'mnist' in args.dataset:
		if args.tasks == 20 and args.hiddens < 256:
			print("Warning! the main paper MLP with 256 neurons for experiment with 20 tasks")
		return MLP(args.hiddens, {'dropout': args.dropout}).to(DEVICE)
	elif 'cifar' in args.dataset:
		return ResNet18(
			config={'input_size': (3, 32, 32), 'dropout': args.dropout, 'classes': int(100 / args.tasks)}).to(DEVICE)
	elif 'imagenet' in args.dataset:
		return ResNet18(
			config={'input_size': (3, 84, 84), 'dropout': args.dropout, 'classes': int(100 / args.tasks)}).to(DEVICE)
	elif 'cub' in args.dataset:
		return ResNet18_CUB(
			config={'input_size': (3, 224, 224), 'dropout': args.dropout, 'classes': int(200 / args.tasks)}).to(DEVICE)
	else:
		raise Exception("Unknown dataset.\n" + "The code supports 'perm-mnist, rot-mnist, and cifar-100.")



def log_metrics(metrics, time, task_id, acc_db, loss_db, p=False):
	"""
	Log accuracy and loss at different times of training
	"""
	if p:
		print('epoch {}, task:{}, metrics: {}'.format(time, task_id, metrics))
	# log to db
	acc = metrics['accuracy']
	loss = metrics['loss']
	loss_db[task_id][time-1] = loss
	acc_db[task_id][time-1] = acc
	return acc_db, loss_db


def save_checkpoint(model, time):
	"""
	Save checkpoints of model paramters
	:param model: pytorch model
	:param time: int
	"""
	# filename = '{directory}/model-{trial}-{time}.pth'.format(directory=EXPERIMENT_DIRECTORY, trial=TRIAL_ID, time=time)
	# torch.save(model.cpu().state_dict(), filename)


def visualize_result(df, filename):
	ax = sns.lineplot(data=df,  dashes=False)
	ax.figure.savefig(filename, dpi=250)
	plt.close()
