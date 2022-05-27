import cv2
import torch
from torch import nn
from torchvision.transforms import transforms
import numpy as np
from util import config


def pred(generator, image):
    transform = transforms.ToTensor()
    tensor = transform(image)[0].unsqueeze(0).unsqueeze(1).to(config.device)
    predicted_image = generator(tensor)
    img = predicted_image.detach().cpu()
    return (img[0] > 0.5).to(torch.float32)


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal(m.weight)
        m.bias.data.fill_(0.01)


def skel(img):
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    ret, img = cv2.threshold(img, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while (not done):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
    return skel

