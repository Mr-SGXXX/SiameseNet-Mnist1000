import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST


class MNIST_1000(Dataset):
    def __init__(self, path, test_flag, siamese_flag=False, transform=None):
        self.test_flag = test_flag
        self.siamese_flag = siamese_flag
        mnist = MNIST(path, not test_flag, transform, download=True)
        if test_flag:
            self.data = [mnist[i] for i in range(len(mnist))]
        else:
            self.data = [mnist[int(i * len(mnist) / 1000)] for i in range(1000)]
        if siamese_flag:
            self.pairs = []
            for i in range(1000):
                for j in range(i + 1, 1000):
                    self.pairs.append((self.data[i][0], self.data[j][0], int(self.data[i][1] != self.data[j][1])))
            random.shuffle(self.pairs)

    def __getitem__(self, item):
        if self.test_flag or not self.siamese_flag:
            return self.data[item]
        else:
            return self.pairs[item]

    def __len__(self):
        if self.test_flag or not self.siamese_flag:
            return len(self.data)
        else:
            return len(self.data) * (len(self.data) - 1) // 2

class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0, p=1):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.p=p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w = img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w))
            # N = np.repeat(N, c, axis=2)
            img = N + img
            img[img > 255] = 255                       # 避免有值超过255而反转
            img = Image.fromarray(img.astype('uint8'))
            return img
        else:
            return img

