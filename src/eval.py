# !/usr/bin/python
# coding: utf-8
from __future__ import print_function
from __future__ import division

import os
from PIL import Image
from tqdm import tqdm
import os
import torch as K
import torch

# import numpy as K

'''
用于分类的指标
'''

epsilon = 1e-10


def recall(ypred, ytrue):
    """
    召回率 哪些小偷被抓了
    :param ytrue torch tensor
    :param ypred torch tensor
    :return float
    """
    tp = K.sum(ypred * ytrue)
    tp_fn = K.sum(ytrue)
    return tp / (tp_fn + epsilon)


def precision(ypred, ytrue):
    """
    准确率 被抓的哪些是对的
    """
    tp = K.sum(ypred * ytrue)
    tp_fp = K.sum(ypred)
    return tp / (tp_fp + epsilon)


def f1(ypred, ytrue, all=False):
    """
    F1
    """
    R = recall(ypred, ytrue)
    P = precision(ypred, ytrue)
    if all:
        return P, R, 2 * P * R / (P + R)
    return 2 * P * R / (P + R + epsilon)


def acc(ypred, ytrue):
    """
    精确度
    """
    # assert (ypred.dim() == 1 and ytrue.dim() == 1)
    cnt = ypred.eq(ytrue).sum()
    return float(cnt) / len(ypred)


def eval_fc(ypred, ytrue, metrics=None):
    """
    统计分类各个指标
    """
    if metrics is None:
        metrics = [recall, precision, f1, acc]
    result = [(str(m.__name__), m(ypred, ytrue)) for m in metrics]
    print(">>> " + '\t'.join(["{}: {:.7f}".format(name, metric) for name, metric in result]))


'''
for segmentation
'''


def load_img(path, gray):
    img = Image.open(path)  ## RGB??
    if gray:
        img = img.convert("L")

    return np.asarray(img, np.uint8)


def eval_seg_dir(rst_dir, eval_file=None):
    filenames = os.listdir(rst_dir)
    gt_file = sorted([f for f in filenames if 'gt' in f])
    rst_file = sorted([f for f in filenames if 'rst' in f])
    if eval_file is None:
        eval_file = 'eval.txt'

    eval_file = os.path.join(rst_dir, eval_file)

    # print(len(gt_file), len(rst_file))
    assert (len(gt_file) == len(rst_file))

    with open(eval_file, 'w') as outfile:
        mean_list = np.zeros((len(gt_file), 4))
        for i in tqdm(range(len(gt_file))):
            # print(gt_file[i])
            gt = load_img(os.path.join(rst_dir, gt_file[i]), True) / 255
            rst = load_img(os.path.join(rst_dir, rst_file[i]), True) / 255

            assert (len(gt.shape) == 2 and len(rst.shape) == 2)
            pa, ma, miu, fiu = eval_pair(rst, gt)
            mean_list[i, :] = [pa, ma, miu, fiu]
            eval_rst = '{}: pixel accuracy {}, mean accuracy {}, mean iu {}, frequency weighted IU {}'.format(
                os.path.basename(rst_dir), pa, ma, miu, fiu)
            outfile.write(eval_rst + "\n")

        outfile.write('{}\n'.format(np.mean(mean_list, axis=0)))

    print(open(eval_file).readlines()[-1])  # print lass value


def eval_pair(image, gt):
    """
    evaluate specified metrics of a batch of images

    params:

    return:
    """
    prob = image
    label = gt

    pa = pixel_accuracy(prob, label)
    ma = mean_accuracy(prob, label)
    miu = mean_IU(prob, label)
    fiu = frequency_weighted_IU(prob, label)
    return pa, ma, miu, fiu


def eval_pair_batch(preds, gts, mean=True, print_result=True):
    """
    evaluate specified metrics of a batch of images

    params:

    return:
    """
    preds, gts = np.squeeze(preds), np.squeeze(gts)

    if np.ndim(preds) == 2:
        preds = np.expand_dims(preds, axis=0)

    if np.ndim(gts) == 2:
        gts = np.expand_dims(gts, axis=0)

    # print(preds.shape, gts.shape)
    results = np.zeros((preds.shape[0], 4))
    for idx, (prob, label) in enumerate(zip(preds, gts)):
        # print(prob.shape, label.shape)
        pa = pixel_accuracy(prob, label)
        ma = mean_accuracy(prob, label)
        miu = mean_IU(prob, label)
        fiu = frequency_weighted_IU(prob, label)
        results[idx, :] = np.asarray([pa, ma, miu, fiu])

    if mean:
        mean_value = np.mean(results, axis=0)
        pa, ma, miu, fiu = mean_value
        if print_result:
            print('>>> pixel acc: {:.7f}, mean acc: {:.7f} miu: {:.7f} fiu: {:.7f}'.format(pa, ma, miu, fiu))

        return mean_value

    return results


'''
Martin Kersner, m.kersner@gmail.com
2015/11/30

Evaluation metrics for image segmentation inspired by
paper Fully Convolutional Networks for Semantic Segmentation.
'''

import numpy as np


def pixel_accuracy(eval_segm, gt_segm):
    '''
    sum_i(n_ii) / sum_i(t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    sum_n_ii = 0
    sum_t_i = 0

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        sum_n_ii += np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        sum_t_i += np.sum(curr_gt_mask)

    if (sum_t_i == 0):
        pixel_accuracy_ = 0
    else:
        pixel_accuracy_ = sum_n_ii / sum_t_i

    return pixel_accuracy_


def mean_accuracy(eval_segm, gt_segm):
    '''
    (1/n_cl) sum_i(n_ii/t_i)
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    accuracy = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)

        if (t_i != 0):
            accuracy[i] = n_ii / t_i

    mean_accuracy_ = np.mean(accuracy)
    return mean_accuracy_


def mean_IU(eval_segm, gt_segm):
    '''
    (1/n_cl) * sum_i(n_ii / (t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    _, n_cl_gt = extract_classes(gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    IU = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        IU[i] = n_ii / (t_i + n_ij - n_ii)

    mean_IU_ = np.sum(IU) / n_cl_gt
    return mean_IU_


def frequency_weighted_IU(eval_segm, gt_segm):
    '''
    sum_k(t_k)^(-1) * sum_i((t_i*n_ii)/(t_i + sum_j(n_ji) - n_ii))
    '''

    check_size(eval_segm, gt_segm)

    cl, n_cl = union_classes(eval_segm, gt_segm)
    eval_mask, gt_mask = extract_both_masks(eval_segm, gt_segm, cl, n_cl)

    frequency_weighted_IU_ = list([0]) * n_cl

    for i, c in enumerate(cl):
        curr_eval_mask = eval_mask[i, :, :]
        curr_gt_mask = gt_mask[i, :, :]

        if (np.sum(curr_eval_mask) == 0) or (np.sum(curr_gt_mask) == 0):
            continue

        n_ii = np.sum(np.logical_and(curr_eval_mask, curr_gt_mask))
        t_i = np.sum(curr_gt_mask)
        n_ij = np.sum(curr_eval_mask)

        frequency_weighted_IU_[i] = (t_i * n_ii) / (t_i + n_ij - n_ii)

    sum_k_t_k = get_pixel_area(eval_segm)

    frequency_weighted_IU_ = np.sum(frequency_weighted_IU_) / sum_k_t_k
    return frequency_weighted_IU_


'''
Auxiliary functions used during evaluation.
'''


def get_pixel_area(segm):
    return segm.shape[0] * segm.shape[1]


def extract_both_masks(eval_segm, gt_segm, cl, n_cl):
    eval_mask = extract_masks(eval_segm, cl, n_cl)
    gt_mask = extract_masks(gt_segm, cl, n_cl)

    return eval_mask, gt_mask


def extract_classes(segm):
    cl = np.unique(segm)
    n_cl = len(cl)

    return cl, n_cl


def union_classes(eval_segm, gt_segm):
    eval_cl, _ = extract_classes(eval_segm)
    gt_cl, _ = extract_classes(gt_segm)

    cl = np.union1d(eval_cl, gt_cl)
    n_cl = len(cl)

    return cl, n_cl


def extract_masks(segm, cl, n_cl):
    h, w = segm_size(segm)
    masks = np.zeros((n_cl, h, w))

    for i, c in enumerate(cl):
        masks[i, :, :] = segm == c

    return masks


def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise

    return height, width


def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")


'''
Exceptions
'''


class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


if __name__ == '__main__':
    eval_fc(torch.rand((10, 10)).view((-1,)), torch.rand((10, 10)).view((-1,)))
