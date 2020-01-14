# coding: utf-8

import numpy as np
from torch.autograd import Variable
import cv2
from skimage import filters
from skimage import morphology as mor
import matplotlib.pyplot as plt


def boundary_from_gt(gt):
    bd = filters.sobel(gt)
    bd = mor.binary_dilation(bd, selem=np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    bd = bd.astype(np.uint8)
    return bd

# visdom helper
def vis_line(data, vis, title, subject='train_loss'):
    x = np.arange(1, len(data) + 1, 1)
    vis.line(data, x, env=subject, opts=dict(title=title))


import os
import cv2


def bmg2png(folder_a, folder_b):
    filenames = os.listdir(folder_a)
    filepaths = [os.path.join(folder_a, name) for name in filenames]
    for filepath, filename in zip(filepaths, filenames):
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(os.path.join(folder_b, filename[:-3] + 'png'), img)


def vis_image(image, vis, title):
    if image.is_cuda:
        image = image.cpu()

    if isinstance(image, Variable):
        image = image.data

    image = image.numpy()
    vis.image(image, env='train-images', opts=dict(title=title))


if __name__ == '__main__':
    # bmg2png('/home/adam/Gits/blur-seg/grid_db/gt', '/home/adam/Gits/blur-seg/grid_db/gt_png')
    boundary_from_gt(cv2.imread(r'E:\Exp\blurcvpr\gt\motion0001.png', cv2.IMREAD_GRAYSCALE)/255)
