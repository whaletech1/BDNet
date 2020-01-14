#  coding: utf-8
from __future__ import print_function

import sys

sys.path.insert(0, '/home/adam/caffe/python')
import caffe
import pickle
import re
from collections import OrderedDict
import torch

class Layer(object):
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.params = []

def caffe2pytorch(protofile, caffemodel, result_filename, include_list=None, exclude_list=None, save_all=False,
                  dump=False, name_pattern=None):
    """
        convert the weights of caffe to torch
    """
    net = caffe.Net(protofile, caffemodel, caffe.TEST)
    params_to_save = net.params  # dict
    layers_to_save = OrderedDict(zip(net._layer_names, net.layers))
    if not save_all and include_list is not None:
        params_to_save = {k: v for k, v in params_to_save.items() if k in include_list}
        layers_to_save = {k: v for k, v in layers_to_save.items() if k in include_list}

    if not save_all and exclude_list is not None:
        params_to_save = {k: v for k, v in params_to_save.items() if k not in exclude_list}
        layers_to_save = {k: v for k, v in layers_to_save.items() if k not in exclude_list}

    layers_list = []
    re_pattern = None if name_pattern is None else re.compile(name_pattern)
    for k, v in layers_to_save.items():
        if re_pattern is None or re_pattern.match(k):
            lr = Layer(k, v.type)
            if lr.type == 'Convolution':
                for p in params_to_save[k]:
                    lr.params.append(p.data)
                    print(lr.name, lr.type, p.data.shape)

            layers_list.append(lr)

    # for lr in layers_list:
    #     print(str(lr))

    if dump:
        # pickle.dump(layers_list, open(result_filename, 'w'))
        torch.save(layers_list, result_filename)

    return params_to_save
    # pass


def caffe2tf():
    """
    convert the weights of caffe to torch
    :return:
    """

    pass


def tf2pytorch():
    """
    convert the weights of tensorflow to pytorch
    :return:
    """

    pass


def prepare_for_class(train_file):
    ## 增加類別
    des_file = train_file[:train_file.rindex('.')] + '_fc.txt'
    lines = open(train_file).readlines()
    with open(des_file, 'w') as des:
        for line in lines:
            if 'motion' in line:
                line = line.strip() + '\tmotion\n'
            elif 'out_of_focus' in line:
                line = line.strip() + '\tout_of_focus\n'

            des.write(line)


if __name__ == '__main__':
    # prepare_for_class('/home/adam/Gits/blur-seg/grid_db/train_pair.txt')
    # prepare_for_class('/home/adam/Gits/blur-seg/grid_db/val_pair.txt')
    prototxt = '/home/adam/Gits/blur-seg/fcns/paper/vgg16/blur_deploy_fcn2s_once.prototxt'
    weights = '/home/adam/Gits/blur-seg/fcns/archive/fcn32s-heavy-pascal.caffemodel'
    # caffe2pytorch(prototxt, weights, 'fcn32s-fc-pascal.pth', include_list=['fc6', 'fc7'], dump=True)
    caffe2pytorch(prototxt, weights, 'fcn32s-fc-pascal-all.pth', dump=True, name_pattern='(conv\s*)|(fc\s*)',
                  save_all=True)
