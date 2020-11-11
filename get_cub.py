import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import torchvision as tv

IMG_MEAN = np.array((103.94,116.78,123.68), dtype=np.float32)


def _CUB_read_img_from_file(data_dir, file_name, img_height, img_width):
    imgs = []
    labels = []

    with open(file_name) as f:
        for line in tqdm(f):
            img_name, img_label = line.split()
            img_file = data_dir.rstrip('\/') + '/' + img_name
            img = cv2.imread(img_file).astype(np.float32)
            img = cv2.resize(img, (img_width, img_height))
            # Convert RGB to BGR
            # img_r, img_g, img_b = np.split(img, 3, axis=2)
            # img = np.concatenate((img_b, img_g, img_r), axis=2)/255.
            # Extract mean
            # img -= IMG_MEAN
            img = (img/255.-0.5)*2
            imgs += [img]
            labels += [int(img_label)]

    # Convert the labels to one-hot
    # y = dense_to_one_hot(np.array(labels))

    return np.array(imgs), np.array(labels)


def _CUB_get_data(data_dir, train_list_file, test_list_file, img_height, img_width):
    """ Reads and parses examples from CUB dataset """

    dataset = dict()
    dataset['train'] = []
    dataset['test'] = []

    # Read train and test files
    train_img, train_label = _CUB_read_img_from_file(data_dir, train_list_file, img_height, img_width)
    test_img, test_label = _CUB_read_img_from_file(data_dir, test_list_file, img_height, img_width)
    return train_img, train_label, test_img, test_label


class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tar'
    filename = 'CUB_200_2011.tar'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        # bounding_box = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'bounding_boxes.txt'),
        #                                sep=' ', names=['img_id', 'x', 'y', 'dx', 'dy'])
        data = images.merge(image_class_labels, on='img_id')
        # data = data.merge(bounding_box, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        # download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r") as tar:
            tar.extractall(path=self.root)

    def get_y(self, idx):
        sample = self.data.iloc[idx]
        return sample.target - 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = cv2.imread(path).astype(np.float32)
        # bbox = self.data.iloc[idx][['x', 'y', 'dx', 'dy']].astype(int)
        # img = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
        img = cv2.resize(img, (224, 224))
        img -= np.array((103.94,116.78,123.68), dtype=np.float32)
        img /= 255.
        # img -= np.array([0.485, 0.456, 0.406])
        # img /= np.array([0.229, 0.224, 0.225])
        return img, target
