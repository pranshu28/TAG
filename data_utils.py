import torchvision
import torchvision.transforms.functional as TorchVisionFunc
from data_utils_2 import *
from tqdm import tqdm
import tarfile
import os
import cv2
import imageio


class MyDataloader(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.images = X/255.
        self.labels = torch.from_numpy(Y)
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return torch.from_numpy(self.images[idx].transpose((2, 0, 1))).float(), self.labels[idx]


def get_nomnist(task_id):
    classes = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    tar_path = "./data/notMNIST_small.tar"
    tmp_path = "./data/tmp"

    img_arr = []
    lab_arr = []

    with tarfile.open(tar_path) as tar:
        tar_root = tar.next().name
        for ind, c in enumerate(classes):
            files = [f for f in tar.getmembers() if f.name.startswith(tar_root+'/'+c)]
            if not os.path.exists(tmp_path):
                os.mkdir(tmp_path)
            for f in files:
                f_obj = tar.extractfile(f)
                try:
                    arr = np.asarray(imageio.imread(f_obj))
                    img = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                    img = cv2.resize(img, (32, 32))
                    img_arr.append(np.asarray(img))
                    lab_arr.append(ind+task_id*len(classes))
                except:
                    continue
    os.rmdir(tmp_path)
    return np.array(img_arr), np.array(lab_arr)


def get_5_datasets(task_id, DATA, batch_size, get_val=False):
    """
    Returns the dataset for a single task of Rotation MNIST dataset
    :param task_id:
    :param batch_size:
    :return:
    """
    if task_id in [0,2]:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),

        ])
    else:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.Lambda(lambda x: x.convert('RGB')),
            torchvision.transforms.ToTensor(),
        ])
    target_transform = torchvision.transforms.Compose([torchvision.transforms.Lambda(lambda y: y+task_id*10)])
    if task_id != 3:
        try:
            train_data = DATA('./data/', train=True, download=True, transform=transforms, target_transform=target_transform)
            test_data = DATA('./data/', train=False, download=True, transform=transforms, target_transform=target_transform)
        except:
            train_data = DATA('./data/SVHN/', split='train', download=True, transform=transforms, target_transform=target_transform)
            test_data = DATA('./data/SVHN/', split='test', download=True, transform=transforms, target_transform=target_transform)
        test_loader = torch.utils.data.DataLoader(test_data,batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    else:
        all_images, all_labels = get_nomnist(task_id)
        dataset_size = len(all_images)
        indices = list(range(dataset_size))
        split = int(np.floor(0.1 * dataset_size))
        np.random.shuffle(indices)
        train_indices, test_indices = indices[split:], indices[:split]
        train_data = MyDataloader(all_images[train_indices], all_labels[train_indices])
        test_data = MyDataloader(all_images[test_indices], all_labels[test_indices])
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
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
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = None
    return train_loader, test_loader, val_loader


def get_5_datasets_tasks(num_tasks, batch_size, get_val=False):
    """
    Returns data loaders for all tasks of rotation MNIST dataset.
    :param num_tasks: number of tasks in the benchmark.
    :param batch_size:
    :return:
    """
    datasets = {}
    data_list = [torchvision.datasets.CIFAR10, torchvision.datasets.MNIST, torchvision.datasets.SVHN, 'notMNIST',  torchvision.datasets.FashionMNIST]
    for task_id, DATA in enumerate(data_list):
        print('Loading Task/Dataset:', task_id)
        train_loader, test_loader, val_loader = get_5_datasets(task_id, DATA, batch_size, get_val=get_val)
        datasets[task_id] = {'train': train_loader, 'test': test_loader, 'val':val_loader}
    return datasets



def get_permuted_mnist(task_id, batch_size):
    """
    Get the dataset loaders (train and test) for a `single` task of permuted MNIST.
    This function will be called several times for each task.

    :param task_id: id of the task [starts from 1]
    :param batch_size:
    :return: a tuple: (train loader, test loader)
    """

    # convention, the first task will be the original MNIST images, and hence no permutation
    if task_id == 1:
        idx_permute = np.array(range(784))
    else:
        idx_permute = torch.from_numpy(np.random.RandomState().permutation(784))
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(lambda x: x.view(-1)[idx_permute] ),
                ])
    target_transform = torchvision.transforms.Compose([torchvision.transforms.Lambda(lambda y: y+(task_id-1)*10)])
    mnist_train = torchvision.datasets.MNIST('./data/', train=True, download=True, transform=transforms, target_transform=target_transform)
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=False, download=True, transform=transforms, target_transform=target_transform),  batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader


def get_permuted_mnist_tasks(num_tasks, batch_size):
    """
    Returns the datasets for sequential tasks of permuted MNIST

    :param num_tasks: number of tasks.
    :param batch_size: batch-size for loaders.
    :return: a dictionary where each key is a dictionary itself with train, and test loaders.
    """
    datasets = {}
    for task_id in range(1, num_tasks+1):
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
    :param task_id:
    :param batch_size:
    :return:
    """
    rotation_degree = (task_id - 1)*per_task_rotation
    rotation_degree -= (np.random.random()*per_task_rotation)

    transforms = torchvision.transforms.Compose([
        RotationTransform(rotation_degree),
        torchvision.transforms.ToTensor(),
        ])
    target_transform = torchvision.transforms.Compose([torchvision.transforms.Lambda(lambda y: y+(task_id-1)*10)])

    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=True, download=True, transform=transforms, target_transform=target_transform), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data/', train=False, download=True, transform=transforms, target_transform=target_transform),  batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader


def get_rotated_mnist_tasks(num_tasks, batch_size):
    """
    Returns data loaders for all tasks of rotation MNIST dataset.
    :param num_tasks: number of tasks in the benchmark.
    :param batch_size:
    :return:
    """
    datasets = {}
    per_task_rotation = {1:360, 2:180, 3: 120, 4: 90, 5:60, 6: 60, 7:45, 8:45, 9:30, 10:30}[num_tasks] if num_tasks<=10 else 10
    print('per_task_rotation =', per_task_rotation)
    for task_id in range(1, num_tasks+1):
        train_loader, test_loader = get_rotated_mnist(task_id, batch_size, per_task_rotation)
        datasets[task_id] = {'train': train_loader, 'test': test_loader}
    return datasets


def get_split_cifar100(task_id, classes, batch_size, cifar_train, cifar_test, get_val=False):
    """
    Returns a single task of split CIFAR-100 dataset
    :param task_id:
    :param batch_size:
    :return:
    """


    start_class = (task_id-1)*classes
    end_class = task_id * classes

    targets_train = torch.tensor(cifar_train.targets)
    target_train_idx = ((targets_train >= start_class) & (targets_train < end_class))

    targets_test = torch.tensor(cifar_test.targets)
    target_test_idx = ((targets_test >= start_class) & (targets_test < end_class))
    train_data = torch.utils.data.dataset.Subset(cifar_train, np.where(target_train_idx==1)[0])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.dataset.Subset(cifar_test, np.where(target_test_idx==1)[0]), batch_size=256)
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
    Returns data loaders for all tasks of split CIFAR-100
    :param num_tasks:
    :param batch_size:
    :return:
    """
    datasets = {}

    # convention: tasks starts from 1 not 0 !
    # task_id = 1 (i.e., first task) => start_class = 0, end_class = 4
    # normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    cifar_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
    cifar_train = torchvision.datasets.CIFAR100('./data/', train=True, download=True, transform=cifar_transforms)
    cifar_test = torchvision.datasets.CIFAR100('./data/', train=False, download=True, transform=cifar_transforms)
    classes = int(100/num_tasks)

    for task_id in range(1, num_tasks+1):
        train_loader, test_loader, val_loader = get_split_cifar100(task_id, classes, batch_size, cifar_train, cifar_test, get_val=get_val)
        datasets[task_id] = {'train': train_loader, 'test': test_loader, 'val': val_loader}
    return datasets
