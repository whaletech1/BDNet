# coding: utf-8
from PIL import Image
import os
import cv2
import sys
import math
from random import randint
from keras.preprocessing.image import flip_axis
from keras.preprocessing.image import ImageDataGenerator as KerasImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import logging
import time
import itertools
from tqdm import tqdm

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# test_file = r'/home/adam/Gits/blur-seg/datasets/raw/src/data/motion0013.jpg'
test_file = r'E:\Exp\blurcvpr\image\motion0013.jpg'


def convert_image(func, parent_dir, src_dir, dst_dir):
    """
    转换图像，bgr->hsv, fft
    """
    file_dir = parent_dir + os.path.sep + src_dir
    filenames = os.listdir(file_dir)
    for filename in tqdm(filenames):
        fullname = '{}/{}'.format(file_dir, filename)
        im = load_img(fullname, data_type="cv")
        data = func(im)
        np.save(parent_dir + os.path.sep + dst_dir +
                os.path.sep + filename[:-4], data)


def flip_im(src, gt, axis=2):
    return flip_axis(src, axis), flip_axis(gt, axis)


def bgr2hsv(bgr_im):
    """
    bgr2hsv BGR颜色空间到HSV颜色空间
    :param bgr_im:
    :return:
    """
    hsv_im = cv2.cvtColor(bgr_im, cv2.COLOR_BGR2HSV)
    return hsv_im


def test_bgr2hsv():
    im = load_img(test_file)
    cv2.imshow("image", im)
    cv2.imshow("hsv", bgr2hsv(im))
    cv2.imshow("H", bgr2hsv(im)[:, :, 0])
    cv2.imshow("S", bgr2hsv(im)[:, :, 1])
    cv2.imshow("V", bgr2hsv(im)[:, :, 2])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def fft_im(src_im):
    """
    fft变换
    :param src: 彩色图像 bgr?
    :return:
    """
    im = np.fft.fft(src_im)
    return im


def test_fft_im():
    im = load_img(test_file)
    cv2.imshow("image", im)
    cv2.imshow("fft", fft_im(im).astype(np.uint8))

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def load_img(image_path,
             gray=False,
             gray_expand=False,
             data_type="cv",
             mean_value=None):
    """
    params:
        tyep: caffe|tf|then
        mean value: bgr form
    return:
        image
    """
    im = None
    if gray:
        im = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    else:
        im = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_COLOR)

    if im is None:
        raise Exception("load image error: {}".format(image_path))

    if mean_value is not None:
        im = im.astype(np.float)
        im -= mean_value

    if data_type in ['tf']:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    if gray_expand:
        im = im[..., np.newaxis]

    if data_type not in ['tf', 'cv']:
        im = np.transpose(im, (2, 0, 1))

    return im


def save_img(full_name, im, cvt_BGR=False, data_type='cv'):
    if data_type not in ['cv', 'tf']:
        im = np.transpose(im, (1, 2, 0))
    if cvt_BGR:
        im = im[:, :, ::-1]
    if im.dtype != np.uint8:
        im = im.astype(np.uint8)
    result = cv2.imwrite(full_name, im)
    if not result:
        raise Exception("save file error: {}".format(full_name))


def crop(image,
         gt,
         mode="grid",
         max_out=10,
         target_size=(256, 256),
         strides=(128, 128)):
    """
    params:
        image: loaded image/ opencv format
        mode: grid|random|grid_append  if grid then ignore
        n_out: number of patches
    return:
        images, gts

    """
    if mode == 'grid':
        patches = np.array(
            [[i, j]
             for i in range(0, image.shape[0] - target_size[0], strides[0])
             for j in range(0, image.shape[1] - target_size[1], strides[1])])
        if max_out is not None and len(patches) > max_out:
            patches = patches[:max_out]

    elif mode == "random":
        a = np.random.randint(
            0, high=(image.shape[0] - target_size[0]), size=(max_out, 1))
        b = np.random.randint(
            0, high=(image.shape[1] - target_size[1]), size=(max_out, 1))
        patches = np.concatenate([a, b], axis=1)

    elif mode == "grid_append":
        # patches = np.array(
        #     [[i, j]
        #      for i in range(0, image.shape[0] - target_size[0], strides[0])
        #      for j in range(0, image.shape[1] - target_size[1], strides[1])])
        patches = [
            (i, j)
            for i in range(0, image.shape[0] - target_size[0], strides[0])
            for j in range(0, image.shape[1] - target_size[1], strides[1])
        ]

        if image.shape[0] % strides[0] > strides[0] / 3:
            patches.extend([
                (image.shape[0] - target_size[0], j)
                for j in range(0, image.shape[1] - target_size[1], strides[1])
            ])

        if image.shape[1] % strides[1] > strides[1] / 3:
            patches.extend([
                (i, image.shape[1] - target_size[1])
                for i in range(0, image.shape[0] - target_size[0], strides[0])
            ])

        if image.shape[0] % strides[0] > strides[0] / 3 and image.shape[
                1] % strides[1] > strides[1] / 3:
            patches.extend([(image.shape[0] - target_size[0],
                             image.shape[1] - target_size[1])])

        # print patches
        patches = list(set(patches))
        patches = np.array(patches)
        if max_out is not None and len(patches) > max_out:
            patches = patches[:max_out]

    images = np.array([
        image[p[0]:p[0] + target_size[0], p[1]:p[1] + target_size[1], :]
        for p in patches
    ])
    gts = np.array([
        gt[p[0]:p[0] + target_size[0], p[1]:p[1] + target_size[1]] / 255
        for p in patches
    ])
    print('crop info, images shape: {}, gts shape: {}, patches shape: {}'.format(
        images.shape, gts.shape, patches.shape))
    return images, gts


def gen_db(root_dir,
           src_dir,
           gt_dir,
           crop_type="grid",
           n_crop=10,
           crop_size=(256, 256),
           crop_strides=(128, 128),
           crop_policy="class_equal",
           with_raw_resize=False,
           save_fmt="bmp",
           split_factor=0.9):
    """
    params:
        root_dir: save
        src_dir: raw src files
        gt_dir: raw gt files
        crop_type: => crop(mode={})
        crop_policy: defalut-do noting; class_equal: result is class equal
        with_raw_resize: resize raw to augment data
        save_fmt: png bmp jpg
    results:
        root_dir/
            srcs/
            gts/
            train.txt
            train_ext.txt
            train_pair.txt
            val.txt (filename without extend)
            val_ext.txt (filename with extend)
            val_pair.txt (with abs path)
    """
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    if not os.path.exists(src_dir) or not os.path.exists(gt_dir):
        raise Exception("src or gt dir doesn't exits")

    single_names = sorted(os.listdir(src_dir))
    short_names = zip(sorted(os.listdir(src_dir)), sorted(os.listdir(gt_dir)))
    full_names = [(os.path.join(src_dir, f[0]), os.path.join(gt_dir, f[1]))
                  for f in short_names]

    n_motion = n_crop
    n_oof = n_crop
    if crop_policy == 'class_equal':
        count_motion = len(
            [f for f in os.listdir(src_dir) if f.startswith('motion')])
        count_oof = len(os.listdir(src_dir)) - count_motion
        radio = max(count_motion, count_oof) / \
            float(min(count_motion, count_oof))
        if count_motion > count_oof:
            n_oof = math.ceil(n_oof * radio)
        else:
            n_motion = math.ceil(n_motion * radio)

        print("motion number:{} out of focus number:{} radio:{} n_oof:{}, n_motion:{}".format(
            count_motion, count_oof, radio, n_oof, n_motion))

    crop_motion_counter = 0
    crop_oof_counter = 0

    for i in range(len(full_names)):
        print('{}: processing {}'.format(i, short_names[i][0]))
        src_path = full_names[i][0]
        gt_path = full_names[i][1]

        im_src = load_img(src_path, data_type='cv')
        im_gt = load_img(gt_path, gray=True, data_type='cv')
        print('load image: {}, load gt: {}'.format(im_src.shape, im_gt.shape))
        if short_names[i][0].startswith('motion'):
            crop_srcs, crop_gts = crop(
                im_src,
                im_gt,
                mode=crop_type,
                max_out=n_motion,
                target_size=crop_size,
                strides=crop_strides)
            crop_motion_counter += crop_srcs.shape[0]

        if short_names[i][0].startswith('out_of'):
            crop_srcs, crop_gts = crop(
                im_src,
                im_gt,
                mode=crop_type,
                max_out=n_oof,
                target_size=crop_size,
                strides=crop_strides)
            crop_oof_counter += crop_srcs.shape[0]

        # print 'crop result, src shape: {}, gt shape:
        # {}'.format(crop_srcs.shape, crop_gts.shape)

        save_imgs(root_dir, "gt", crop_gts, short_names[i][0][:-4], save_fmt,
                  "")
        save_imgs(root_dir, "src", crop_srcs, short_names[i][0][:-4], save_fmt,
                  "")

        if with_raw_resize:
            im_src_resize = cv2.resize(im_src, crop_size)
            im_gt_resize = cv2.resize(im_gt, crop_size)
            save_imgs(root_dir, "src", [im_src_resize], short_names[i][0][:-4],
                      save_fmt, "(raw)")
            save_imgs(root_dir, "gt", [im_gt_resize / 255],
                      short_names[i][0][:-4], save_fmt, "(raw)")
            if short_names[i][0].startswith('motion'):
                crop_motion_counter += 1

            if short_names[i][0].startswith('out_of'):
                crop_oof_counter += 1

        print('> crop result oof: ', crop_srcs.shape[0])
        print('> crop result motion: ', crop_srcs.shape[0])

    print('>>crop summary: crop_motion: {}, crop_oof: {}'.format(
        crop_motion_counter, crop_oof_counter))

    n_train = int(math.ceil(len(single_names) * split_factor))

    np.random.shuffle(single_names)
    seleced_files = single_names[:n_train]
    seleced_files = [f[:-4] for f in seleced_files]

    crop_files = sorted(os.listdir(os.path.join(root_dir, "src")))
    is_train = np.array(
        map(lambda x: x[:x.rfind('_')] in seleced_files, crop_files),
        dtype=np.bool)
    crop_files = np.array(crop_files)

    train_list = list(crop_files[is_train])
    val_list = list(crop_files[np.invert(is_train)])

    train_motion = 0
    train_oof = 0
    for train_file in train_list:
        if train_file.startswith('motion'):
            train_motion += 1
        if train_file.startswith('out'):
            train_oof += 1

    val_motion = 0
    val_oof = 0
    for val_file in val_list:
        if val_file.startswith('motion'):
            val_motion += 1
        if val_file.startswith('out'):
            val_oof += 1
    print('>> summary: train_motion {}, train_oof {}, val_motion: {}, val_oof {}'.format(
        train_motion, train_oof, val_motion, val_oof))

    with open(os.path.join(root_dir, "train.txt"), 'w') as channel:
        channel.write('\n'.join(list(crop_files[is_train])))

    with open(os.path.join(root_dir, "train_pair.txt"), 'w') as channel:
        channel.write('\n'.join(
            map(lambda x: os.path.join(root_dir, 'src', x) + '\t' + os.path.join(root_dir, 'gt', x),
                crop_files[is_train])))

    with open(os.path.join(root_dir, "val.txt"), 'w') as channel:
        channel.write('\n'.join(list(crop_files[np.invert(is_train)])))

    with open(os.path.join(root_dir, "val_pair.txt"), 'w') as channel:
        channel.write('\n'.join(
            map(lambda x: os.path.join(root_dir, 'src', x) + '\t' + os.path.join(root_dir, 'gt', x),
                crop_files[np.invert(is_train)])))


def check_or_makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_imgs(root_dir, sub_dir, images, prefix, fmt, index, data_type="cv"):
    """
    save image:
    params: root_dir-root dir;
    """

    check_or_makedirs(os.path.join(root_dir, sub_dir))
    i = 0
    for image in images:
        image_path = "{}_{}{}.{}".format(
            os.path.join(root_dir, sub_dir, prefix), index, i, fmt)
        ok = cv2.imwrite(image_path, image)
        if not ok:
            raise Exception("write image error: {}".format(image_path))
        i = i + 1


if __name__ == '__main__':
    # gen_db("blurdb", "datasets/raw/src", "datasets/raw/cgt", n_crop=5, crop_type='random', crop_size=(256,256))
    gen_db(
        "/home/adam/Gits/blur-seg/grid_db",
        "/home/adam/Gits/blur-seg/datasets/raw/src",
        "/home/adam/Gits/blur-seg/datasets/raw/cgt",
        n_crop=None,
        crop_type='grid_append',
        crop_size=(256, 256),
        crop_strides=(128, 128),
        crop_policy="default",
        with_raw_resize=True,
        save_fmt="bmp")
