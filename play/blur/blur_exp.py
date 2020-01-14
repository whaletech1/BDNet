# coding: utf-8

import cv2
from skimage import filters, data
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import entropy
from numpy.linalg import norm

"""
测试下梯度直方图分布的相关性和模糊程度的关系
"""


def KL(p, q):
    return entropy(p, q)


def SYM_KL(p, q):
    return 0.5 * (KL(p, q) + KL(q, p))


def JSD(p, q):
    _m = 0.5 * (p + q)
    return 0.5 * (entropy(p, _m) + entropy(q, _m))


def make_same_shape(a, b):
    _len = max(len(a), len(b))
    _a = np.pad(a, (0, _len - len(a)), 'constant')
    _b = np.pad(b, (0, _len - len(b)), 'constant')
    return _a, _b


def exp():
    def grad_hist(img, show=False, label=''):
        grad_x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
        hist_x, _ = np.histogram(np.ravel(np.abs(grad_x)), range=(0, 256), bins=50)
        hist_x = hist_x / (img.shape[0] * img.shape[1])
        plt.plot(hist_x, label=label, marker='*')
        plt.xlim((0, 30))
        plt.ylim((0, 1.0))
        plt.tight_layout()
        if show:
            plt.show()

    # img = data.camera()
    img = cv2.imread(r'D:\Gits\blur-detection\imgs\house.jpg')
    print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.blur(img, ksize=(17, 17))
    # img = cv2.blur(img, ksize=(17, 17))
    grad_hist(img, label='blurred')
    resized = cv2.resize(img, (img.shape[0] // 2, img.shape[1] // 2))
    # resized = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=1.5)
    # resized = cv2.blur(img, ksize=(11, 11))
    grad_hist(resized, label='downsampled')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    exp()
