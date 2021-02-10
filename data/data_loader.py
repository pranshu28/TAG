from data.data_utils import *
from data.data_utils_2 import *


def get_benchmark_data_loader(args):
	"""
    Returns the benchmark loader based on MNIST:
    get_permuted_mnist_tasks, or get_rotated_mnist_tasks
    :param args:
    :return: a function which when called, returns all tasks
    """
	if args.dataset == 'perm-mnist' or args.dataset == 'permuted-mnist':
		return get_permuted_mnist_tasks
	elif args.dataset == 'rot-mnist' or args.dataset == 'rotation-mnist':
		return get_rotated_mnist_tasks
	else:
		raise Exception("Unknown dataset.\n" + "The code supports 'perm-mnist and rot-mnist.")


def get_data_loaders(args, grid_search=False):
	"""
	Get data loaders for the given dataset
	"""
	print("Loading {} tasks for {}".format(args.tasks, args.dataset))
	if args.dataset in ['cifar100','cifar10']:
		tasks = get_split_cifar100_tasks(args.tasks, args.batch_size, grid_search)
		train_loaders, test_loaders = [tasks[i]['train'] for i in tasks], [tasks[i]['test'] for i in tasks]
		val_loaders = [tasks[i]['val'] for i in tasks]
		args.classes = 100
	elif args.dataset == 'imagenet':
		train_loaders, test_loaders, val_loaders = [CLDataLoader(elem, args, train=t)
		                                            for elem, t in zip(get_miniimagenet(args, grid_search), [True, False, False])]
		args.classes = 100
	elif args.dataset == 'cub':
		train_loaders, test_loaders, val_loaders = [CLDataLoader(elem, args, train=t)
		                                            for elem, t in zip(get_split_cub(args, grid_search), [True, False, False])]
		args.classes = 200
	elif args.dataset == '5data':
		tasks = get_5_datasets_tasks(args.tasks, args.batch_size, grid_search)
		train_loaders, test_loaders = [tasks[i]['train'] for i in tasks], [tasks[i]['test'] for i in tasks]
		val_loaders = [tasks[i]['val'] for i in tasks]
		args.classes = 50
	else:
		tasks = get_benchmark_data_loader(args)(args.tasks, args.batch_size)
		train_loaders, test_loaders = [tasks[i]['train'] for i in tasks], [tasks[i]['test'] for i in tasks]
		val_loaders = None
		args.classes = 10
	print("loaded all tasks!")
	return train_loaders, test_loaders, val_loaders


