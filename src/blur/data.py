# coding: utf-8
from __future__ import print_function

import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as ts

import os
from PIL import Image
import numpy as np
import random

import cv2

from src.blur.util import boundary_from_gt

import torchvision.transforms.functional as F

def load_img(img_path):
    try:
        return Image.open(img_path)
    except Exception as e:
        print("error in opening image: ", img_path)


def load_gray_img(img_path):
    try:
        return Image.open(img_path).convert("L")
    except Exception as e:
        print("error in opening image: ", img_path)

class PILDictDataset(data.Dataset):
    def __init__(self, data_file,
                 data_contain_fc=False,
                 transform=None,
                 img_transform=None,
                 label_transform=None,
                 with_label=True,
                 only_select=None,
                 **kwargs):
        super(PILDictDataset, self).__init__()

        data = torch.load(data_file)
        if data_contain_fc:
            self.images, self.labels, self.tags = data['images'], data['gts'], data['tags']
            if only_select is not None:
                tag = None
                if only_select == 'motion':
                    tag = 0
                elif only_select == 'out_of_focus':
                    tag = 1

                if tag is not None:
                    images = []
                    labels = []
                    for _tag, _image, _label in zip(self.tags, self.images, self.labels):
                        if _tag == tag:
                            images.append(_image)
                            labels.append(_label)

                    self.images = images  # set images
                    self.labels = labels
                    self.tags = list([tag, ] * len(self.images))
                    assert len(self.images) == len(self.labels)
        else:
            self.images, self.labels = data['images'], data['gts']

        self.with_fc = data_contain_fc
        self.transform = transform
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.with_label = with_label
        self.with_edge = kwargs['with_edge']
        self.only_edge = kwargs['only_edge']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = self.images[idx]

        label = None
        edge = None
        if self.with_label:
            label = self.labels[idx]
            if self.with_edge or self.only_edge:
                edge = boundary_from_gt(label)
                edge = Image.fromarray(edge)

        # transform
        if self.transform is not None:
            if edge is not None:
                img, label, edge = self.transform(img, label, edge)
            else:
                img, label = self.transform(img, label)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            label = self.label_transform(label)
            if edge is not None:
                edge= self.label_transform(edge)

        if self.with_fc:
            if self.only_edge:
                return img, edge, self.tags[idx]

            if self.with_edge:
                return img, label, self.tags[idx], edge

            return img, label, self.tags[idx]

        if self.only_edge:
            return img, edge

        if self.with_edge:
            return img, label, edge

        return img, label

    @staticmethod
    def instance(data_file, test=True, with_fc=False, only_select=None, **kwargs):
        trans = None if test else compose_transform()
        return PILDictDataset(data_file, data_contain_fc=with_fc, transform=trans, img_transform=img_transform(),
                              label_transform=label_transform(), only_select=only_select, **kwargs)


class SegDataset(data.Dataset):
    """
    :arg
    """

    def __init__(self, train_list_file, root_dir=None, transform=None,
                 img_transform=None, label_transform=None, with_label=True, **kwargs):
        """

        :param train_list:
        :param transform:
        :param target_transform:
        :param lazy_loading:
        """

        super(SegDataset, self).__init__()
        self.train_list_file = train_list_file
        self.transform = transform
        self.root_dir = root_dir
        self.with_label = with_label
        self.img_transform = img_transform
        self.label_transform = label_transform
        # self.with_edge = kwargs['with_edge']
        self.with_edge = False
        # self.only_edge = kwargs['only_edge']
        self.only_edge = False

        try:
            train_list = list(open(train_list_file).readlines())
        except Exception as e:
            print('error in opening file: ', train_list_file, e.value)

        cnt_motion = sum([1 if line.find('motion') != -1 else 0 for line in train_list])
        if not cnt_motion == 0:
            # if kwargs['weight_of_motion'] is not None:
            #     radio = kwargs['weight_of_motion']
            # else:
            #     radio = (len(train_list) - cnt_motion) / cnt_motion
            radio = (len(train_list) - cnt_motion) / cnt_motion
            print('sample radio', radio)

            self.sample_weight = [radio if line.find('motion') != -1 else 1 for line in train_list]

        train_list = [m.split() for m in train_list]
        self.train_list = [f[0] for f in train_list]
        if with_label:
            self.label_list = [f[1] for f in train_list]

    def get_weight(self):
        return self.sample_weight

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):

        img_path = self.train_list[idx]
        if self.root_dir is not None:
            img_path = os.path.join(self.root_dir, img_path)

        img = self.load_img(img_path)

        label = None
        edge = None
        if self.with_label:
            label_path = self.label_list[idx]
            if self.root_dir is not None:
                label_path = os.path.join(self.root_dir, label_path)

            label = self.load_gray_img(label_path)
            if self.with_edge or self.only_edge:
                edge = boundary_from_gt(label)
                edge = Image.fromarray(edge)

        # transform
        if self.with_label and self.transform is not None:
            if edge is not None:
                img, label, edge = self.transform(img, label, edge)
            else:
                img, label = self.transform(img, label)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            label = self.label_transform(label)
            if edge is not None:
                edge= self.label_transform(edge)

        if self.with_label:
            if self.only_edge:
                return img, edge

            if self.with_edge:
                return img, label, edge

            return img, label

        else:
            if self.with_edge or self.only_edge:
                return img, edge

            return img

    def load_img(self, img_path):
        return load_img(img_path)

    def load_gray_img(self, img_path):
        return load_gray_img(img_path)

    def loader(self, batch_size, shuffle=True, num_workers=2):
        return data.DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers);

    @staticmethod
    def instance(train_list_file, **kwargs):
        return SegDataset(train_list_file, transform=compose_transform(),
                          label_transform=label_transform(), **kwargs)


from torch.utils.data import sampler


class WeigtableDataset(object):
    def __data_len_list(self):
        raise NotImplementedError

    def __weight_list(self):
        raise NotImplementedError


class WeigtedSampler(sampler.Sampler):
    def __init__(self, data_source):
        super(WeigtedSampler, self).__init__(data_source)
        assert isinstance(data_source, WeigtableDataset)
        self.data_source = data_source
        self.data_len_list = data_source.__data_len_list
        self.perm_len_list = [iter(torch.randperm(e).long()) for e in self.data_len_list]

        weight_list = data_source.__weight_list
        np_weights = np.asarray(weight_list)
        self.p_weights = np_weights / np.sum(np_weights)  ## normalize

        self.choice = np.random.choice(len(self.data_len_list), self.__len__(), False, p=self.p_weights)

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        for c in self.choice:
            yield self.perm_len_list[c].__next__()  ## shold be a problem


class SeperatableSegDataset(data.Dataset, WeigtableDataset):
    """
    可选择数据样本的数据集
    """

    def __init__(self, train_list_file, root_dir=None, transform=None,
                 img_transform=None, label_transform=None, with_label=True,
                 only_select=None, iter_radio_motion_by_oof=None):
        super(SeperatableSegDataset, self).__init__()

        self.train_list_file = train_list_file
        self.transform = transform
        self.root_dir = root_dir
        self.with_label = with_label
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.sample_random_radio = None

        data_list = list(open(train_list_file).readlines())
        motion_list = [e for e in data_list if e.find('motion') != -1]
        oof_list = [e for e in data_list if e.find('out_of_focus') != -1]

        ## 控制每个batch 的采样比
        if iter_radio_motion_by_oof is not None:
            if iter_radio_motion_by_oof == 'auto':
                self.sample_random_radio = float(len(oof_list)) / (len(motion_list) + len(oof_list))
            else:
                self.sample_random_radio = iter_radio_motion_by_oof / (iter_radio_motion_by_oof + 1.0)

        if only_select is not None:
            if only_select == 'motion':
                data_list = motion_list
            elif only_select == 'out_of_focus':
                data_list = oof_list

            else:
                raise Exception(
                    'invalid parameter for only select, expect motion || out_of_foucs || none, but received {}'.format(
                        only_select))

        data_list = [m.split() for m in data_list]
        self.train_list = [f[0] for f in data_list]
        if with_label:
            self.label_list = [f[1] for f in data_list]

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):
        img_path = self.train_list[idx]
        if self.root_dir is not None:
            img_path = os.path.join(self.root_dir, img_path)

        img = load_img(img_path)

        label = None
        if self.with_label:
            label_path = self.label_list[idx]
            if self.root_dir is not None:
                label_path = os.path.join(self.root_dir, label_path)

            label = load_gray_img(label_path)

        # transform
        if self.transform is not None:
            img, label = self.transform(img, label)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.label_transform is not None:
            label = self.label_transform(label)

        return img, label

    def make_loader(self, *args, **kwargs):
        return data.DataLoader(self, *args, **kwargs)

    @staticmethod
    def instance(*args, **kwargs):
        return SeperatableSegDataset(*args, **kwargs, transform=compose_transform(), img_transform=img_transform(),
                                     label_transform=label_transform())


class SegDatasetFC(SegDataset):
    dict_class = {
        'motion': 0,
        'out_of_focus': 1
    }

    def __init__(self, train_list_file, root_dir=None, transform=None,
                 img_transform=None, label_transform=None, with_label=True):
        super(SegDatasetFC, self).__init__(train_list_file, root_dir=root_dir, transform=transform,
                                           img_transform=img_transform, label_transform=label_transform,
                                           with_label=with_label)
        train_list = list(open(train_list_file).readlines())
        self.tag_list = [self.dict_class[line.split()[2].strip()] for line in train_list]

    def __getitem__(self, idx):
        img, label = super(SegDatasetFC, self).__getitem__(idx)
        tag = self.tag_list[idx]
        return img, label, tag

    @staticmethod
    def instance(train_list_file):
        return SegDatasetFC(train_list_file, transform=compose_transform(), img_transform=img_transform(),
                            label_transform=label_transform())


#####################################################################
#   preprocessing
####################################################################

class LabelToTensor(object):
    def __call__(self, label):
        return torch.from_numpy(np.array(label)).long()


class ImageToHSV(object):
    def __call__(self, image):
        if isinstance(image, Image.Image):
            return image.convert('HSV')

        return None


class RandomCrop(object):
    def __init__(self, size):
        super(RandomCrop, self).__init__()
        self.target_size = size

    def __call__(self, *inputs):
        w, h = inputs[0].size
        th, tw = self.target_size
        if w == tw and h == th:
            return inputs

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return [x.crop((x1, y1, x1 + tw, y1 + th)) for x in inputs]

class CenterCrop(object):
    def __init__(self, target_size):
        super(CenterCrop, self).__init__()
        self.target_size = target_size

    def __call__(self, img, label):
        w, h = img.size
        th, tw = self.target_size
        if w <= tw and h <= th:
            return img, label

        x1 = w // 2 - tw // 2
        y1 = h // 2 - th // 2
        img, label = img.crop((x1, y1, x1 + tw, y1 + th)), label.crop((x1, y1, x1 + tw, y1 + th))
        return img, label


class SegResizeTo(object):
    def __init__(self, target_size, mode=Image.BILINEAR):
        super(SegResizeTo, self).__init__()
        self.target_size = target_size
        self.mode = mode

    def __call__(self, img, label):
        img, label = img.resize(self.target_size, resample=self.mode), label.resize(self.target_size,
                                                                                    resample=self.mode)
        return img, label


class RandomFlip(object):
    def __init__(self, direction='H'):
        super(RandomFlip, self).__init__()
        self.direction = direction

    def __call__(self, *inputs):
        if self.direction == "H":
            if random.random() < 0.5:
                return [x.transpose(Image.FLIP_TOP_BOTTOM) for x in  inputs]

        elif self.direction == "W":
            if random.random() < 0.5:
                return [x.transpose(Image.FLIP_LEFT_RIGHT) for x in  inputs]

        return inputs


class SegRandomResizedCrop(ts.RandomResizedCrop):
    def __init__(self, *args, **kwargs):
        super(SegRandomResizedCrop, self).__init__(*args, **kwargs)

    def __call__(self, *inputs):
        """
        :param img:  PIL image
        :param label:
        :return:
        """
        i, j, h, w = self.get_params(inputs[0], self.scale, self.ratio)
        return [F.resized_crop(x, i, j, h, w, self.size, self.interpolation) for x in inputs]


class SegCompose(object):
    def __init__(self, transform):
        super(SegCompose, self).__init__()
        self.transforms = transform

    def __call__(self, *inputs):
        for t in self.transforms:
            inputs = t(*inputs)

        return inputs


def label_transform():
    return ts.Compose([
        LabelToTensor(),
    ])


## todo refctor
def nuceli_img_transform():
    return ts.Compose([
        # ImageToHSV(),
        # ts.RandomGrayscale(p=1),
        # ts.ColorJitter(brightness=0.1, contrast=0.3, saturation=0.1),
        ts.ToTensor(),
        # ts.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def img_transform():
    return ts.Compose([
        ts.ToTensor(),
        # ts.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def compose_transform():
    return SegCompose([
        RandomCrop(size=(224, 224)),
        # SegRandomResizedCrop(256),
        # RandomFlip(direction="H"),
        RandomFlip(direction="W"),
    ])


##################################### test ##############################################
def test_loader():
    def imshow(img, label=False):
        # img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        if label:
            npimg = npimg.astype(np.uint8) * 255

        print(npimg.shape)
        subfix = 'img'
        if label:
            subfix = 'label'

        cv2.imshow("hello " + subfix, np.transpose(npimg, (1, 2, 0)))
        cv2.waitKey(0)

    train_list_file = 'list_fc.txt'
    data_train = SegDatasetFC(train_list_file, transform=compose_transform(), img_transform=img_transform(),
                              label_transform=label_transform())
    data_train_loader = data_train.loader(5)
    data_train_iter = iter(data_train_loader)
    images, labels, tags = data_train_iter.next()
    labels = torch.unsqueeze(labels, 1)
    # print(labels.size())
    imshow(torchvision.utils.make_grid(images))
    imshow(torchvision.utils.make_grid(labels), True)
    print(tags, len(tags))
    print("after show")


def test_wegith_sampler():
    from torch.utils.data import sampler
    weight = list([1, ] * 30)
    weight[:10] = list([3, ] * 10)
    weight_sampler = sampler.WeightedRandomSampler(weight, num_samples=len(weight))
    batch_sampler = sampler.BatchSampler(weight_sampler, batch_size=4, drop_last=False)
    for indices in batch_sampler:
        print(indices)


if __name__ == '__main__':
    test_loader()
    test_wegith_sampler()
