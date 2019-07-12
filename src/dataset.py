import torch
from torch.utils.data import Dataset
import glob
from PIL import Image
from torchvision import transforms
import os
import pickle
import scipy.io as sio
import numpy as np


class MNISTToSVHN(Dataset):

    def __init__(self, mnist_img_dir, svhn_img_dir):
        self.train_mnist_img, self.train_mnist_labels = self._load_mnist(
                mnist_img_dir)
        self.train_svhn_img, self.train_svhn_labels = self._load_svhn(
                svhn_img_dir)
        self.mnist_img_list = glob.glob(mnist_img_dir + "/*.jpg")  # 100
        self.svhn_img_list = glob.glob(svhn_img_dir + "/*.jpg")  # 10
        self.svhn_transforms = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.mnist_transforms = transforms.Compose([
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):

        mnist_img = self.mnist_transforms(
                torch.Tensor(self.train_mnist_img[index]))
        mnist_img = mnist_img.expand(3, 32, 32)
        mnist_label = self.train_mnist_labels[index]
        svhn_img = self.svhn_transforms(
                torch.Tensor(self.train_svhn_img[index]))
        svhn_label = self.train_svhn_labels[index]
        return svhn_img.float(), svhn_label, mnist_img.float(), mnist_label

    def _load_svhn(self, image_dir, split='train'):
        print('loading svhn image dataset..')

        image_file = 'train_32x32.mat' if split == 'train' else 'test_32x32.mat'

        image_dir = os.path.join(image_dir, image_file)
        svhn = sio.loadmat(image_dir)
        images = np.transpose(svhn['X'], [3, 2, 0, 1]) / 127.5 - 1
        labels = svhn['y'].reshape(-1)
        labels[np.where(labels == 10)] = 0
        print('finished loading svhn image dataset..!')
        return images, labels

    def _load_mnist(self, image_dir, split='train'):
        print('loading mnist image dataset..')
        image_file = 'train.pkl' if split == 'train' else 'test.pkl'
        image_dir = os.path.join(image_dir, image_file)
        with open(image_dir, 'rb') as f:
            mnist = pickle.load(f)
        images = mnist['X'] / 127.5 - 1
        labels = mnist['y']
        print('finished loading mnist image dataset..!')
        return images, labels

    def __len__(self):
        return 50000
