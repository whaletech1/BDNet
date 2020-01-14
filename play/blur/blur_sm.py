# coding: utf-8
import numpy as np
import pandas as pd
import cv2
from operator import itemgetter
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from skimage.filters import threshold_otsu
from pathos.multiprocessing import Pool, cpu_count

"""
对缩小算法的重新实现
"""


def min_max(m):
    _min, _max = np.min(m), np.max(m)
    _m = (m - _min) / (_max - _min)
    return _m


def cvt_img(img):
    return (img * 255).astype(np.uint8)


def pool_processor(func, iter):
    pool = Pool(cpu_count())
    pool.map(func, iter)
    # pool.join()
    pool.close()


def ssim(x, y, c1=0.01 * 255 * 0.01 * 255, c2=0.03 * 255 * 0.03 * 255, c3=None, alpha=1, beta=1, gamma=1):
    """
    ssim指标
    :param x: x image gray
    :param y: image
    :param c1: 0.01 * 255 * 0.01 * 255
    :param c2: 0.03 * 255 * 0.03 * 255
    :param c3: C2 / 2
    :param alpha: 幂指数 default 1
    :param beta:
    :param gamma:
    :return:
    """
    if c3 is None:
        c3 = c2 / 2
    mux, muy = cv2.mean(x)[0], cv2.mean(y)[0]
    square_x, square_y, xy = x * x, y * y, x * y
    sigma_x = np.sqrt(cv2.mean(square_x)[0] - mux * mux)
    sigma_y = np.sqrt(cv2.mean(square_y)[0] - muy * muy)
    sigma_xy = cv2.mean(xy)[0] - mux * muy

    lxy = (2 * mux * muy + c1) / (mux * mux + muy * muy + c1)
    cxy = (2 * sigma_x * sigma_y + c2) / (sigma_x * sigma_x + sigma_y * sigma_y + c2)
    sxy = (sigma_xy + c3) / (sigma_x * sigma_y + c3)
    ans = pow(lxy, alpha) * pow(cxy, beta) * pow(sxy, gamma)
    return ans


def nrss(img, blur, block_size=8, N=32):
    """
    NRSS指标
    :param img:  gray image
    :param blur: blurred gray image
    :param block_size:
    :return:
    """
    gx, gy = cv2.Sobel(img, cv2.CV_16SC1, 1, 0), cv2.Sobel(img, cv2.CV_16SC1, 0, 1)
    gxy = cv2.addWeighted(np.abs(gx), 0.5, np.abs(gy), 0.5, 0.0, dtype=cv2.CV_32FC1)

    bgx, bgy = cv2.Sobel(blur, cv2.CV_16SC1, 1, 0), cv2.Sobel(blur, cv2.CV_16SC1, 0, 1)
    bgxy = cv2.addWeighted(np.abs(bgx), 0.5, np.abs(bgy), 0.5, 0.0, dtype=cv2.CV_32FC1)

    _rows, _cols = img.shape
    _step = block_size // 2
    block_stds = [((idx_row, idx_col), np.std(gxy[idx_row - block_size:idx_row, idx_col - block_size:idx_col])) for
                  idx_row in range(block_size, _rows + 1, _step) for idx_col in range(block_size, _cols + 1, _step)]
    block_stds = sorted(block_stds, key=itemgetter(1))  # sort ascending
    block_stds = block_stds[-(min(N, len(block_stds) - 1)):]  # remove last one

    ssim_scores = []
    for (idx_row, idx_col), _ in block_stds:
        img_block = gxy[idx_row - block_size:idx_row, idx_col - block_size:idx_col]
        blurred_block = bgxy[idx_row - block_size:idx_row, idx_col - block_size:idx_col]
        score = ssim(img_block, blurred_block)
        ssim_scores.append(score)

    return sum(ssim_scores) / len(ssim_scores)


def test_sm(filename, sm_scale=2, patch_size=32):
    """
    原始sm方法的核心实现
    :param filename:
    :param sm_scale:
    :param patch_size:
    :return:
    """
    img = cv2.imread(filename)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = img.shape
    sm_h, sm_w, sm_patch_size = h // sm_scale, w // sm_scale, patch_size // sm_scale
    resized = cv2.resize(img, (sm_w, sm_h))

    img_blurred = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=3, borderType=cv2.BORDER_REFLECT)
    sm_blurred = cv2.GaussianBlur(resized, ksize=(0, 0), sigmaX=3, borderType=cv2.BORDER_REFLECT)

    result_size = (sm_h - sm_patch_size + 1, sm_w - sm_patch_size + 1)
    sm_nrss_score = np.zeros(result_size)
    raw_nrss_score = np.zeros(result_size)
    # for idx_row in tqdm(range(sm_patch_size, sm_h + 1)):
    for idx_row in range(sm_patch_size, sm_h + 1):
        for idx_col in range(sm_patch_size, sm_w + 1):
            sm_patch = resized[idx_row - sm_patch_size:idx_row, idx_col - sm_patch_size:idx_col]
            sm_blurred_patch = sm_blurred[idx_row - sm_patch_size:idx_row, idx_col - sm_patch_size:idx_col]
            idx_raw_row, idx_raw_col = idx_row * sm_scale, idx_col * sm_scale
            raw_patch = img[(idx_raw_row - patch_size):idx_raw_row, (idx_raw_col - patch_size):idx_raw_col]
            raw_blurred_patch = img_blurred[(idx_raw_row - patch_size):idx_raw_row,
                                (idx_raw_col - patch_size):idx_raw_col]
            # print(idx_row, idx_col, idx_raw_row, idx_raw_col, sm_patch.shape, raw_patch.shape)
            sm_nrss, raw_nrss = nrss(sm_patch, sm_blurred_patch), nrss(raw_patch, raw_blurred_patch)
            sm_nrss_score[idx_row - sm_patch_size, idx_col - sm_patch_size] = sm_nrss
            raw_nrss_score[idx_row - sm_patch_size, idx_col - sm_patch_size] = raw_nrss

    nrss_diff = sm_nrss_score - raw_nrss_score
    nrss_abs_diff = np.abs(sm_nrss_score - raw_nrss_score)
    return nrss_diff, sm_nrss_score, raw_nrss_score, nrss_abs_diff


def single_sm(filename, sm_scale=2, patch_size=32):
    """
    快速算法，no diff
    :param filename:
    :param sm_scale:
    :param patch_size:
    :return:
    """
    img = cv2.imread(filename)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = img.shape
    sm_h, sm_w, sm_patch_size = h // sm_scale, w // sm_scale, patch_size // sm_scale
    if sm_scale != 1:
        resized = cv2.resize(img, (sm_w, sm_h))
    else:
        resized = img

    sm_blurred = cv2.GaussianBlur(resized, ksize=(0, 0), sigmaX=3, borderType=cv2.BORDER_REFLECT)

    result_size = (sm_h - sm_patch_size + 1, sm_w - sm_patch_size + 1)
    sm_nrss_score = np.zeros(result_size)
    for idx_row in range(sm_patch_size, sm_h + 1):
        for idx_col in range(sm_patch_size, sm_w + 1):
            sm_patch = resized[idx_row - sm_patch_size:idx_row, idx_col - sm_patch_size:idx_col]
            sm_blurred_patch = sm_blurred[idx_row - sm_patch_size:idx_row, idx_col - sm_patch_size:idx_col]
            sm_nrss = nrss(sm_patch, sm_blurred_patch)
            sm_nrss_score[idx_row - sm_patch_size, idx_col - sm_patch_size] = sm_nrss

    return sm_nrss_score


def time_single_sm():
    filename = r'E:\Blur\blurdetect_cvpr14\BlurDatasetImage\image\motion0133.jpg'
    t = 20
    _start = datetime.now()
    for i in range(t):
        single_sm(filename, 2, 32)
    _end = datetime.now()
    _duration = (_end - _start).total_seconds()
    print("total time {}s, per iter time {}s".format(_duration, _duration / float(t)))

    _start = datetime.now()
    for i in range(t):
        single_sm(filename, 1, 32)
    _end = datetime.now()
    _duration = (_end - _start).total_seconds()
    print("total time {}s, per iter time {}s".format(_duration, _duration / float(t)))


def post_process(img):
    """
    后处理方法
    :param img:
    :return:
    """
    if type(img) == str:
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    ## upsample

    ## guided filter

    ## otsu
    thresh = threshold_otsu(img)
    binary = img >= thresh
    binary = binary.astype(np.uint8) * 255
    return binary


def batch_post_process(src_dir, dst_dir):
    """
    多线程批量执行后处理任务
    :param src_dir:
    :param dst_dir:
    :return:
    """

    def p(name):
        filename = os.path.join(src_dir, name)
        binary = post_process(filename)
        cv2.imwrite(os.path.join(dst_dir, name), binary)

    names = os.listdir(src_dir)
    pool_processor(p, names)


def do_blur():
    """
    多线程并行运行核心算法
    :return:
    """
    if os.name == 'nt':
        base_dir = r'E:\Exp\blurcvpr\image'
        dst_dir = r'E:\Exp\blurcvpr\result_sm_b64'

    else:
        base_dir = r'/home/adam/Datasets/BlurDataset/image'
        # dst_dir = r'/home/adam/Datasets/BlurDataset/result_sm'
        # dst_dir = r'/home/adam/Datasets/BlurDataset/result_sm_b128'
        dst_dir = r'/home/adam/Datasets/BlurDataset/result_sm_b64'

    def p(img_name):
        print(img_name)
        _start = datetime.now()
        result_diff, result_sm, result_raw, nrss_abs_diff = test_sm(os.path.join(base_dir, img_name), patch_size=64)
        ok = cv2.imwrite(os.path.join(dst_dir, 'diff_' + img_name), cvt_img(min_max(result_diff)))
        cv2.imwrite(os.path.join(dst_dir, 'result_sm_' + img_name), cvt_img(min_max(result_sm)))
        cv2.imwrite(os.path.join(dst_dir, 'result_raw_' + img_name), cvt_img(min_max(result_raw)))
        cv2.imwrite(os.path.join(dst_dir, 'diff_abs_' + img_name), cvt_img(min_max(nrss_abs_diff)))
        _end = datetime.now()
        print('time cost', _end - _start)

    pool_processor(p, sorted(os.listdir(base_dir)))


if __name__ == '__main__':
    # do_blur()
    # batch_post_process(r'E:\Blur\filtered_results\result_sm', r'E:\Blur\binary_results\result_sm')
    # batch_post_process(r'E:\Blur\filtered_results\result_raw', r'E:\Blur\binary_results\result_raw')
    # batch_post_process(r'E:\Blur\filtered_results\result_diff', r'E:\Blur\binary_results\result_diff')
    # batch_post_process(r'E:\Blur\filtered_results\result_diff_abs', r'E:\Blur\binary_results\result_diff_abs')

    time_single_sm() # 统计执行时间
