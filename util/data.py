import os
import random

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from util.config import inp_shape
from util.edges import get_edges
import numpy as np

from util.tools import skel


class PixToPixDataset(Dataset):
    def __init__(self, root_dir, edges_dir=None):
        self.root_dir = root_dir
        self.edges_dir = edges_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.list_files[idx])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, inp_shape, interpolation=cv2.INTER_AREA)
        if self.edges_dir:
            img_edges_name = os.path.join(self.edges_dir, self.list_files[idx])
            image_edges = cv2.imread(img_edges_name)
            image_edges = cv2.resize(image_edges, inp_shape, interpolation=cv2.INTER_AREA)
            image_edges = cv2.cvtColor(image_edges, cv2.COLOR_BGR2GRAY)
            image_edges = skel(image_edges)
            p = int(random.random() > 0.5)
            transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(p=p),
                    transforms.ToTensor()])
            # Convert the image to PyTorch tensor
            y = transform(image)
            x = transform(image_edges)
            x = (x > 0.5).to(torch.float32)
        else:
            transform = transforms.ToTensor()

            # Convert the image to PyTorch tensor
            y = transform(image)
            x = transform(get_edges(image))

        return x, y
