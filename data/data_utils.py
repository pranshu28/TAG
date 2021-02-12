import torchvision
import torchvision.transforms.functional as TorchVisionFunc
import tarfile
import os
import cv2
import imageio
import numpy as np
import torch
from torchvision import datasets, transforms
from tqdm import tqdm

from sklearn.model_selection import train_test_split
import pickle

import numpy as np
import torch
from torchvision import datasets, transforms
from tqdm import tqdm

from sklearn.model_selection import train_test_split
import pickle


class XYDataset(torch.utils.data.Dataset):
	"""
    Image pre-processing
    """

	def __init__(self, x, y, **kwargs):
		self.x, self.y = x, y

		# this was to store the inverse permutation in permuted_mnist
		# so that we could 'unscramble' samples and plot them
		for name, value in kwargs.items():
			setattr(self, name, value)

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		x, y = self.x[idx], self.y[idx]

		if type(x) != torch.Tensor:
			# mini_imagenet
			# we assume it's a path --> load from file
			x = self.transform(x)
			y = torch.Tensor(1).fill_(y).long().squeeze()
		else:
			x = x.float() / 255.
			y = y.long()

		# for some reason mnist does better \in [0,1] than [-1, 1]
		if self.source == 'mnist':
			return x, y
		else:
			return (x - 0.5) * 2, y


class CLDataLoader(object):
	"""
    Create data loader for the given task dataset
    """

	def __init__(self, datasets_per_task, args, train=True):
		bs = args.batch_size if train else 256

		self.datasets = datasets_per_task
		self.loaders = [
			torch.utils.data.DataLoader(x, batch_size=bs, shuffle=True, drop_last=train, num_workers=0)
			for x in self.datasets]

	def __getitem__(self, idx):
		return self.loaders[idx]

	def __len__(self):
		return len(self.loaders)


def get_split_cub(args, get_val=False):
	"""
    Import CUB dataset and split it into multiple tasks with disjoint set of classes
    Implementation is based on the one provided by:
        Aljundi, Rahaf, et al. "Online continual learning with maximally interfered retrieval."
        arXiv preprint arXiv:1908.04742 (2019).
    :param args: Arguments for model/data configuration
    :param get_val: Get validation set for grid search
    :return: Train, test and validation data loaders
    """
	args.n_classes = 200
	args.n_classes_per_task = args.n_classes
	args.use_conv = True
	args.n_classes_per_task = int(args.n_classes / args.tasks)
	args.input_size = [3, 224, 224]
	args.input_type = 'continuous'
	IMG_MEAN = np.array((103.94, 116.78, 123.68), dtype=np.float32)

	import cv2
	def _CUB_read_img_from_file(data_dir, file_name, img_height, img_width):
		count = 0
		imgs = []
		labels = []

		with open(file_name) as f:
			for line in f:
				img_name, img_label = line.split()
				img_file = data_dir.rstrip('\/') + '/' + img_name
				img = cv2.imread(img_file).astype(np.float32)
				# HWC -> WHC, compatible with caffe weights
				# img = np.transpose(img, [1, 0, 2])
				img = cv2.resize(img, (img_width, img_height))
				# Convert RGB to BGR
				img_r, img_g, img_b = np.split(img, 3, axis=2)
				img = np.concatenate((img_b, img_g, img_r), axis=2)
				# Extract mean
				img -= IMG_MEAN

				imgs += [img]
				labels += [int(img_label)]
				count += 1

				if count % 1000 == 0:
					print('Finish reading ', count)

		return np.array(imgs), np.array(labels)

	# all_data, all_label = _CUB_read_img_from_file('data/CUB_200_2011/images', 'data/CUB_200_2011/images.txt', 224, 224)
	train_img, train_label = _CUB_read_img_from_file('data/CUB_200_2011/images', 'data/CUB_200_2011/CUB_train_list.txt',
	                                                 224, 224)
	test_img, test_label = _CUB_read_img_from_file('data/CUB_200_2011/images', 'data/CUB_200_2011/CUB_test_list.txt',
	                                               224, 224)
	print(train_img.shape, test_img.shape)
	train_ds, test_ds = [], []
	current_train, current_test = None, None

	cat = lambda x, y: np.concatenate((x, y), axis=0)
	for i in range(args.n_classes):
		class_indices = np.argwhere(train_label == i).reshape(-1)
		class_test_indices = np.argwhere(test_label == i).reshape(-1)
		data_train = train_img[class_indices]
		label_train = train_label[class_indices]

		data_test, label_test = test_img[class_test_indices], test_label[class_test_indices]

		if current_train is None:
			current_train, current_test = (data_train, label_train), (data_test, label_test)
		else:
			current_train = cat(current_train[0], data_train), cat(current_train[1], label_train)
			current_test = cat(current_test[0], data_test), cat(current_test[1], label_test)
		if i % args.n_classes_per_task == (args.n_classes_per_task - 1):
			train_ds += [current_train]
			test_ds += [current_test]
			current_train, current_test = None, None

	transform = transforms.Compose([transforms.ToTensor()])
	# build masks
	masks = []
	task_ids = [None for _ in range(args.tasks)]
	for task, task_data in enumerate(train_ds):
		labels = np.unique(task_data[1])  # task_data[1].unique().long()
		assert labels.shape[0] == args.n_classes_per_task
		mask = torch.zeros(args.n_classes).to(args.device)
		mask[labels] = 1
		masks += [mask]
		task_ids[task] = labels

	task_ids = torch.from_numpy(np.stack(task_ids)).to(args.device).long()
	test_ds = map(lambda x, y: XYDataset(x[0], x[1],
	                                     **{'source': 'cub', 'mask': y, 'task_ids': task_ids, 'transform': transform}),
	              test_ds, masks)
	if get_val:
		train_ds, val_ds = make_valid_from_train(train_ds)
		val_ds = map(lambda x, y: XYDataset(x[0], x[1], **{'source': 'cub', 'mask': y, 'task_ids': task_ids,
		                                                   'transform': transform}), val_ds, masks)
	else:
		val_ds = test_ds
	train_ds = map(lambda x, y: XYDataset(x[0], x[1],
	                                      **{'source': 'cub', 'mask': y, 'task_ids': task_ids, 'transform': transform}),
	               train_ds, masks)
	return train_ds, test_ds, val_ds


def get_miniimagenet(args, get_val=False):
	"""
    Import mini-imagenet dataset and split it into multiple tasks with disjoint set of classes
    Implementation is based on the one provided by:
        Aljundi, Rahaf, et al. "Online continual learning with maximally interfered retrieval."
        arXiv preprint arXiv:1908.04742 (2019).
    :param args: Arguments for model/data configuration
    :param get_val: Get validation set for grid search
    :return: Train, test and validation data loaders
    """
	args.use_conv = True
	args.n_classes = 100
	# if args.multi == 1:
	#     args.tasks = 1
	args.n_classes_per_task = args.n_classes
	# else:
	#     args.tasks = 20
	args.n_classes_per_task = 5
	args.input_size = (3, 84, 84)

	transform = transforms.Compose([
		# transforms.Resize(84),
		# transforms.CenterCrop(84),
		transforms.ToTensor(),
	])
	for i in ['train', 'test', 'val']:
		file = open("data/mini_imagenet/mini-imagenet-cache-" + i + ".pkl", "rb")
		file_data = pickle.load(file)
		data = file_data["image_data"]
		if i == 'train':
			main_data = data.reshape([64, 600, 84, 84, 3])
		else:
			app_data = data.reshape([(20 if i == 'test' else 16), 600, 84, 84, 3])
			main_data = np.append(main_data, app_data, axis=0)
	all_data = main_data.reshape((60000, 84, 84, 3))
	all_label = np.array([[i] * 600 for i in range(100)]).flatten()

	train_ds, test_ds = [], []
	current_train, current_test = None, None

	cat = lambda x, y: np.concatenate((x, y), axis=0)

	for i in range(args.n_classes):
		class_indices = np.argwhere(all_label == i).reshape(-1)
		class_data = all_data[class_indices]
		class_label = all_label[class_indices]
		split = int(0.8 * class_data.shape[0])

		data_train, data_test = class_data[:split], class_data[split:]
		label_train, label_test = class_label[:split], class_label[split:]

		if current_train is None:
			current_train, current_test = (data_train, label_train), (data_test, label_test)
		else:
			current_train = cat(current_train[0], data_train), cat(current_train[1], label_train)
			current_test = cat(current_test[0], data_test), cat(current_test[1], label_test)

		if i % args.n_classes_per_task == (args.n_classes_per_task - 1):
			train_ds += [current_train]
			test_ds += [current_test]
			current_train, current_test = None, None

	# build masks
	masks = []
	task_ids = [None for _ in range(20)]
	for task, task_data in enumerate(train_ds):
		labels = np.unique(task_data[1])  # task_data[1].unique().long()
		assert labels.shape[0] == args.n_classes_per_task
		mask = torch.zeros(args.n_classes).to(args.device)
		mask[labels] = 1
		masks += [mask]
		task_ids[task] = labels

	task_ids = torch.from_numpy(np.stack(task_ids)).to(args.device).long()

	test_ds = map(lambda x, y: XYDataset(x[0], x[1], **{'source': 'mini_imagenet', 'mask': y, 'task_ids': task_ids,
	                                                    'transform': transform}), test_ds, masks)
	if get_val:
		train_ds, val_ds = make_valid_from_train(train_ds)
		val_ds = map(lambda x, y: XYDataset(x[0], x[1], **{'source': 'mini_imagenet', 'mask': y, 'task_ids': task_ids,
		                                                   'transform': transform}), val_ds, masks)
	else:
		val_ds = test_ds
	train_ds = map(lambda x, y: XYDataset(x[0], x[1], **{'source': 'mini_imagenet', 'mask': y, 'task_ids': task_ids,
	                                                     'transform': transform}), train_ds, masks)

	return train_ds, test_ds, val_ds


def make_valid_from_train(dataset, cut=0.9):
	"""
    Split training data to get validation set
    :param dataset: Training dataset
    :param cut: Percentage of dataset to be kept for training purpose
    """
	tr_ds, val_ds = [], []
	for task_ds in dataset:
		x_t, y_t = task_ds

		# shuffle before splitting
		perm = torch.randperm(len(x_t))
		x_t, y_t = x_t[perm], y_t[perm]

		split = int(len(x_t) * cut)
		x_tr, y_tr = x_t[:split], y_t[:split]
		x_val, y_val = x_t[split:], y_t[split:]

		tr_ds += [(x_tr, y_tr)]
		val_ds += [(x_val, y_val)]

	return tr_ds, val_ds


class MyDataloader(torch.utils.data.Dataset):
	def __init__(self, X, Y):
		self.images = X / 255.
		self.labels = torch.from_numpy(Y)

	def __len__(self):
		return len(self.images)

	def __getitem__(self, idx):
		return torch.from_numpy(self.images[idx].transpose((2, 0, 1))).float(), self.labels[idx]


def get_nomnist(task_id):
	"""
    Parses and returns the downloaded notMNIST dataset
    """
	classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
	tar_path = "./data/notMNIST_small.tar"
	tmp_path = "./data/tmp"

	img_arr = []
	lab_arr = []

	with tarfile.open(tar_path) as tar:
		tar_root = tar.next().name
		for ind, c in enumerate(classes):
			files = [f for f in tar.getmembers() if f.name.startswith(tar_root + '/' + c)]
			if not os.path.exists(tmp_path):
				os.mkdir(tmp_path)
			for f in files:
				f_obj = tar.extractfile(f)
				try:
					arr = np.asarray(imageio.imread(f_obj))
					img = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
					img = cv2.resize(img, (32, 32))
					img_arr.append(np.asarray(img))
					lab_arr.append(ind + task_id * len(classes))
				except:
					continue
	os.rmdir(tmp_path)
	return np.array(img_arr), np.array(lab_arr)


def get_5_datasets(task_id, DATA, batch_size, get_val=False):
	"""
    Returns the data loaders for a single task of 5-dataset
    :param task_id: Current task id
    :param DATA: Dataset class from torchvision
    :param batch_size: Batch size
    :param get_val: Get validation set for grid search
    :return: Train, test and validation data loaders
    """
	if task_id in [0, 2]:
		transforms = torchvision.transforms.Compose([
			torchvision.transforms.ToTensor(),

		])
	else:
		transforms = torchvision.transforms.Compose([
			torchvision.transforms.Resize(32),
			torchvision.transforms.Lambda(lambda x: x.convert('RGB')),
			torchvision.transforms.ToTensor(),
		])
	target_transform = torchvision.transforms.Compose([torchvision.transforms.Lambda(lambda y: y + task_id * 10)])

	# All datasets except notMNIST (task_id=3) are available in torchvision
	if task_id != 3:
		try:
			train_data = DATA('./data/', train=True, download=True, transform=transforms,
			                  target_transform=target_transform)
			test_data = DATA('./data/', train=False, download=True, transform=transforms,
			                 target_transform=target_transform)
		except:
			# Slighly different way to import SVHN
			train_data = DATA('./data/SVHN/', split='train', download=True, transform=transforms,
			                  target_transform=target_transform)
			test_data = DATA('./data/SVHN/', split='test', download=True, transform=transforms,
			                 target_transform=target_transform)
		test_loader = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4,
		                                          pin_memory=True)
	else:
		all_images, all_labels = get_nomnist(task_id)
		dataset_size = len(all_images)
		indices = list(range(dataset_size))
		split = int(np.floor(0.1 * dataset_size))
		np.random.shuffle(indices)
		train_indices, test_indices = indices[split:], indices[:split]
		train_data = MyDataloader(all_images[train_indices], all_labels[train_indices])
		test_data = MyDataloader(all_images[test_indices], all_labels[test_indices])
		test_loader = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4,
		                                          pin_memory=True)
	if get_val:
		dataset_size = len(train_data)
		indices = list(range(dataset_size))
		split = int(np.floor(0.1 * dataset_size))
		np.random.shuffle(indices)
		train_indices, val_indices = indices[split:], indices[:split]
		train_dataset = torch.utils.data.Subset(train_data, train_indices)
		val_dataset = torch.utils.data.Subset(train_data, val_indices)
		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
		val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256)
	else:
		train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4,
		                                           pin_memory=True)
		val_loader = None
	return train_loader, test_loader, val_loader


def get_5_datasets_tasks(num_tasks, batch_size, get_val=False):
	"""
    Returns data loaders for all tasks of 5-dataset.
    :param num_tasks: Total number of tasks
    :param batch_size: Batch-size for training data
    :param get_val: Get validation set for grid search
    """
	datasets = {}
	data_list = [torchvision.datasets.CIFAR10,
	             torchvision.datasets.MNIST,
	             torchvision.datasets.SVHN,
	             'notMNIST',
	             torchvision.datasets.FashionMNIST]
	for task_id, DATA in enumerate(data_list):
		print('Loading Task/Dataset:', task_id)
		train_loader, test_loader, val_loader = get_5_datasets(task_id, DATA, batch_size, get_val=get_val)
		datasets[task_id] = {'train': train_loader, 'test': test_loader, 'val': val_loader}
	return datasets


def get_permuted_mnist(task_id, batch_size):
	"""
    Get the dataset loaders (train and test) for a `single` task of permuted MNIST.
    This function will be called several times for each task.
    :param task_id: Current task id
    :param batch_size: Batch size
    :return: Train and test data loaders
    """

	# convention, the first task will be the original MNIST images, and hence no permutation
	if task_id == 1:
		idx_permute = np.array(range(784))
	else:
		idx_permute = torch.from_numpy(np.random.RandomState().permutation(784))
	transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
	                                             torchvision.transforms.Lambda(lambda x: x.view(-1)[idx_permute]),
	                                             ])
	target_transform = torchvision.transforms.Compose([torchvision.transforms.Lambda(lambda y: y + (task_id - 1) * 10)])
	mnist_train = torchvision.datasets.MNIST('./data/', train=True, download=True, transform=transforms,
	                                         target_transform=target_transform)
	train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, num_workers=4, pin_memory=True,
	                                           shuffle=True)
	test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=False, download=True,
	                                                                     transform=transforms,
	                                                                     target_transform=target_transform),
	                                          batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

	return train_loader, test_loader


def get_permuted_mnist_tasks(num_tasks, batch_size):
	"""
    Returns the datasets for sequential tasks of permuted MNIST
    :param num_tasks: Total number of tasks
    :param batch_size: Batch-size for training data
    """
	datasets = {}
	for task_id in range(1, num_tasks + 1):
		train_loader, test_loader = get_permuted_mnist(task_id, batch_size)
		datasets[task_id] = {'train': train_loader, 'test': test_loader}
	return datasets


class RotationTransform:
	"""
    Rotation transforms for the images in `Rotation MNIST` dataset.
    """

	def __init__(self, angle):
		self.angle = angle

	def __call__(self, x):
		return TorchVisionFunc.rotate(x, self.angle, fill=(0,))


def get_rotated_mnist(task_id, batch_size, per_task_rotation=10):
	"""
    Returns the dataset for a single task of Rotation MNIST dataset
    :param task_id: Current task id
    :param batch_size: Batch size
    :param per_task_rotation: Rotation different between each task
    :return: Train and test data loaders
    """
	rotation_degree = (task_id - 1) * per_task_rotation
	rotation_degree -= (np.random.random() * per_task_rotation)

	transforms = torchvision.transforms.Compose([
		RotationTransform(rotation_degree),
		torchvision.transforms.ToTensor(),
	])
	target_transform = torchvision.transforms.Compose([torchvision.transforms.Lambda(lambda y: y + (task_id - 1) * 10)])

	train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=True, download=True,
	                                                                      transform=transforms,
	                                                                      target_transform=target_transform),
	                                           batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=False, download=True,
	                                                                     transform=transforms,
	                                                                     target_transform=target_transform),
	                                          batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

	return train_loader, test_loader


def get_rotated_mnist_tasks(num_tasks, batch_size):
	"""
    Returns data loaders for all tasks of rotation MNIST dataset.
    :param num_tasks: Total number of tasks
    :param batch_size: Batch-size for training data
    """
	datasets = {}
	per_task_rotation = {1: 360, 2: 180, 3: 120, 4: 90, 5: 60, 6: 60, 7: 45, 8: 45, 9: 30, 10: 30}[
		num_tasks] if num_tasks <= 10 else 10
	print('per_task_rotation =', per_task_rotation)
	for task_id in range(1, num_tasks + 1):
		train_loader, test_loader = get_rotated_mnist(task_id, batch_size, per_task_rotation)
		datasets[task_id] = {'train': train_loader, 'test': test_loader}
	return datasets


def get_split_cifar100(task_id, classes, batch_size, cifar_train, cifar_test, get_val=False):
	"""
    Returns a single task of Split-CIFAR100 dataset
    :param task_id: Current task id
    :param classes: Number of classes per task
    :param batch_size: Batch size
    :param cifar_train: CIFAR100 training data
    :param cifar_test: CIFAR100 test data
    :param get_val: Get validation set for grid search
    :return: Train, test and validation data loaders
    """
	start_class = (task_id - 1) * classes
	end_class = task_id * classes

	targets_train = torch.tensor(cifar_train.targets)
	target_train_idx = ((targets_train >= start_class) & (targets_train < end_class))

	targets_test = torch.tensor(cifar_test.targets)
	target_test_idx = ((targets_test >= start_class) & (targets_test < end_class))
	train_data = torch.utils.data.dataset.Subset(cifar_train, np.where(target_train_idx == 1)[0])
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(
		torch.utils.data.dataset.Subset(cifar_test, np.where(target_test_idx == 1)[0]), batch_size=256)
	if get_val:
		dataset_size = len(train_loader.dataset)
		indices = list(range(dataset_size))
		split = int(np.floor(0.1 * dataset_size))
		np.random.shuffle(indices)
		train_indices, val_indices = indices[split:], indices[:split]
		train_dataset = torch.utils.data.Subset(train_data, train_indices)
		val_dataset = torch.utils.data.Subset(train_data, val_indices)
		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
		val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256)
	else:
		val_loader = None
	return train_loader, test_loader, val_loader


def get_split_cifar100_tasks(num_tasks, batch_size, get_val=False):
	"""
    Returns data loaders for all tasks of Split-CIFAR100
    :param num_tasks: Total number of tasks
    :param batch_size: Batch-size for training data
    :param get_val: Get validation set for grid search
    """
	datasets = {}

	# convention: tasks starts from 1 not 0 !
	# task_id = 1 (i.e., first task) => start_class = 0, end_class = 4
	# normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
	cifar_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])
	cifar_train = torchvision.datasets.CIFAR100('./data/', train=True, download=True, transform=cifar_transforms)
	cifar_test = torchvision.datasets.CIFAR100('./data/', train=False, download=True, transform=cifar_transforms)
	classes = int(100 / num_tasks)

	for task_id in range(1, num_tasks + 1):
		train_loader, test_loader, val_loader = get_split_cifar100(task_id, classes, batch_size, cifar_train,
		                                                           cifar_test, get_val=get_val)
		datasets[task_id] = {'train': train_loader, 'test': test_loader, 'val': val_loader}
	return datasets
