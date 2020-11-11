import numpy as np
import torch
from torchvision import datasets, transforms
from tqdm import tqdm

from sklearn.model_selection import train_test_split
import pickle

""" Template Dataset with Labels """


class XYDataset(torch.utils.data.Dataset):
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


""" Template Dataset for Continual Learning """
class CLDataLoader(object):
	def __init__(self, datasets_per_task, args, train=True):
		bs = args.batch_size if train else 64

		self.datasets = datasets_per_task
		self.loaders = [
			torch.utils.data.DataLoader(x, batch_size=bs, shuffle=True, drop_last=train, num_workers=0)
			for x in self.datasets]

	def __getitem__(self, idx):
		return self.loaders[idx]

	def __len__(self):
		return len(self.loaders)


def get_split_cub_(args):
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
	train_img, train_label = _CUB_read_img_from_file('data/CUB_200_2011/images', 'data/CUB_200_2011/CUB_train_list.txt', 224, 224)
	test_img, test_label = _CUB_read_img_from_file('data/CUB_200_2011/images', 'data/CUB_200_2011/CUB_test_list.txt', 224, 224)
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

	# train_ds, val_ds = make_valid_from_train(train_ds)
	train_ds = map(lambda x, y: XYDataset(x[0], x[1], **{'source': 'cub', 'mask': y, 'task_ids': task_ids, 'transform': transform}), train_ds, masks)
	# val_ds = map(lambda x, y: XYDataset(x[0], x[1] **{'source': 'cub', 'mask': y, 'task_ids': task_ids, 'transform': transform}), val_ds, masks)
	test_ds = map(lambda x, y: XYDataset(x[0], x[1], **{'source': 'cub', 'mask': y, 'task_ids': task_ids, 'transform': transform}), test_ds, masks)

	return train_ds, test_ds


def get_miniimagenet(args):
	args.use_conv = True
	args.n_classes = 100
	# if args.multi == 1:
	# 	args.tasks = 1
	args.n_classes_per_task = args.n_classes
	# else:
	# 	args.tasks = 20
	args.n_classes_per_task = 5
	args.input_size = (3, 84, 84)

	transform = transforms.Compose([
		# transforms.Resize(84),
		# transforms.CenterCrop(84),
		transforms.ToTensor(),
	])
	for i in ['train', 'test', 'val']:
		file = open("data/imagenet/mini-imagenet-cache-" + i + ".pkl", "rb")
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

	# train_ds, val_ds = make_valid_from_train(train_ds)
	train_ds = map(lambda x, y: XYDataset(x[0], x[1], **{'source': 'imagenet', 'mask': y, 'task_ids': task_ids, 'transform': transform}), train_ds, masks)
	# val_ds = map(lambda x, y: XYDataset(x[0], x[1], **{'source': 'imagenet', 'mask': y, 'task_ids': task_ids, 'transform': transform}), val_ds, masks)
	test_ds = map(lambda x, y: XYDataset(x[0], x[1], **{'source': 'imagenet', 'mask': y, 'task_ids': task_ids, 'transform': transform}), test_ds, masks)

	return train_ds, test_ds


def make_valid_from_train(dataset, cut=0.95):
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


# def get_mini_imagenet(task_id, classes, batch_size, train_X, test_X, train_Y, test_Y):
# 	"""
# 	Returns a single task of split CIFAR-100 dataset
# 	:param task_id:
# 	:param batch_size:
# 	:return:
# 	"""
#
# 	start_class = (task_id - 1) * classes
# 	end_class = task_id * classes
#
# 	# train_X = torch.tensor(train_X)
# 	targets_train = torch.tensor(train_Y)
# 	target_train_idx = ((targets_train >= start_class) & (targets_train < end_class))
#
# 	# test_X = torch.tensor(test_X)
# 	targets_test = torch.tensor(test_Y)
# 	target_test_idx = ((targets_test >= start_class) & (targets_test < end_class))
#
# 	train_loader = torch.utils.data.DataLoader(
# 		torch.utils.data.dataset.Subset(tuple(zip(train_X,train_Y)), np.where(target_train_idx == 1)[0]), batch_size=batch_size,
# 		shuffle=True)
# 	test_loader = torch.utils.data.DataLoader(
# 		torch.utils.data.dataset.Subset(tuple(zip(test_X,test_Y)), np.where(target_test_idx == 1)[0]), batch_size=batch_size)
#
# 	return train_loader, test_loader


# def get_mini_imagenet_tasks(num_tasks, batch_size):
# 	"""
# 	Returns data loaders for all tasks of split CUB
# 	:param num_tasks:
# 	:param batch_size:
# 	:return:
# 	"""
# 	datasets = {}
#
# 	# transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])
# 	for i in ['train', 'test', 'val']:
# 		file = open("data/imagenet/mini-imagenet-cache-" + i + ".pkl", "rb")
# 		file_data = pickle.load(file)
# 		data = file_data["image_data"]
# 		if i == 'train':
# 			main_data = data.reshape([64, 600, 84, 84, 3])
# 		else:
# 			app_data = data.reshape([(20 if i == 'test' else 16), 600, 84, 84, 3])
# 			main_data = np.append(main_data, app_data, axis=0)
# 	all_data = (main_data.reshape((60000, 84, 84, 3))/255. - 0.5) *2
# 	# all_data -= np.array((0.406, 0.456, 0.485), dtype=np.float32)  # -> images between 0 and 1
# 	# all_data /= np.array((0.225, 0.224, 0.229), dtype=np.float32)
# 	all_label = np.array([[i] * 600 for i in range(100)]).flatten()
#
# 	train_X, test_X, train_Y, test_Y = train_test_split(all_data, all_label, test_size=0.2)
# 	classes = int(100/num_tasks)
#
# 	for task_id in range(1, num_tasks + 1):
# 		train_loader, test_loader = get_mini_imagenet(task_id, classes, batch_size, train_X, test_X, train_Y, test_Y)
# 		datasets[task_id] = {'train': train_loader, 'test': test_loader}
# 	return datasets

def get_mini_imagenet(task_id, classes, batch_size, all_data, all_label):
	"""
	Returns a single task of split CIFAR-100 dataset
	:param task_id:
	:param batch_size:
	:return:
	"""

	start_class = (task_id - 1) * classes
	end_class = task_id * classes
	target_idx = ((all_label >= start_class) & (all_label < end_class))
	train_X, test_X, train_Y, test_Y = train_test_split(all_data[target_idx], all_label[target_idx], test_size=0.2)
	train_loader = torch.utils.data.DataLoader(tuple(zip(train_X, train_Y)), batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(tuple(zip(test_X, test_Y)), batch_size=batch_size)
	return train_loader, test_loader


def get_mini_imagenet_tasks(num_tasks, batch_size):
	"""
	Returns data loaders for all tasks of split CUB
	:param num_tasks:
	:param batch_size:
	:return:
	"""
	datasets = {}

	# transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), ])
	for i in ['train', 'test', 'val']:
		file = open("data/imagenet/mini-imagenet-cache-" + i + ".pkl", "rb")
		file_data = pickle.load(file)
		data = file_data["image_data"]
		if i == 'train':
			main_data = data.reshape([64, 600, 84, 84, 3])
		else:
			app_data = data.reshape([(20 if i == 'test' else 16), 600, 84, 84, 3])
			main_data = np.append(main_data, app_data, axis=0)
	all_data = (main_data.reshape((60000, 84, 84, 3)) / 255.)  # - 0.5) *2
	all_data -= np.array((0.406, 0.456, 0.485), dtype=np.float32)  # -> images between 0 and 1
	all_data /= np.array((0.225, 0.224, 0.229), dtype=np.float32)
	all_label = np.array([[i] * 600 for i in range(100)]).flatten()
	classes = int(100 / num_tasks)

	for task_id in range(1, num_tasks + 1):
		train_loader, test_loader = get_mini_imagenet(task_id, classes, batch_size, all_data, all_label)
		datasets[task_id] = {'train': train_loader, 'test': test_loader}
	return datasets
