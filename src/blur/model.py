# coding: utf-8
from __future__ import print_function

import sys

from src.blur.loss import *

sys.path.append('..')

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import torch
# from src.prepare import caffe2pytorch
import os
import pickle
import numpy as np
from functools import partial
import unittest


def init_xavier(weight, bias=None):
    """
    使用xavier方式初始化  weight, bias
    :param weight:
    :param bias:
    :return:
    """
    nn.init.xavier_uniform(weight, gain=np.sqrt(2.0))
    if bias is not None:
        nn.init.constant(bias, 0.1)


def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


def init_layers(func, *args):
    """
    初始化所有层
    :param args:
    :return:
    """
    # print('init layers')
    for seq in args:
        if isinstance(seq, nn.ConvTranspose2d):
            print('init deconv layer')
            m, k, h, w = seq.weight.size()
            _w = upsample_filt(h)
            weight = np.zeros((m, k, h, w), dtype=np.float32)
            weight[range(m), range(k), :, :] = _w
            seq.weight.data = torch.from_numpy(weight)

        elif isinstance(seq, nn.Conv2d):
            func(seq.weight, seq.bias)

        elif isinstance(seq, nn.Module):
            modules = [m[1] for m in seq.named_children()]
            init_layers(func, *modules)


def init_from_caffe(layer_list_target, caffe_info_list):
    """
    :param layer_list_target: pytorch caffe model
    :param caffe_info_list: Layer list defined in prepare
    :return:
    """
    for m, layer in zip(layer_list_target, caffe_info_list):
        if isinstance(m, nn.Conv2d) and layer.name == 'Convolution':
            m.weight.data = layer.params[0]
            m.bias.data = layer.params[1]


def center_crop(x, target):
    _, _, th, tw = target.size()
    _, _, h, w = x.size()

    if th > h and tw > w:
        h1 = (th - h) // 2
        h2 = th - h - h1
        w1 = (tw - w) // 2
        w2 = tw - w - w1
        pad = (h1, h2, w1, w2)
        return F.pad(x, pad=pad)

    elif th == h and tw == w:
        return x

    else:
        h1 = (h - th) // 2
        w1 = (w - tw) // 2
        x = x[:, :, h1:h1 + th, w1:w1 + tw]
        return x


def list_center_crop(x_list, target):
    results = []
    for x in x_list:
        results.append(center_crop(x, target))
    return results


##############################################
## model
##############################################

class FCN8s(nn.Module):
    def __init__(self, n_input_channel=3,
                 n_classes=2,
                 pretrained=False,
                 none_classifier=False,
                 init_weight_with_caffe=None):
        """

        :param n_input_channel:
        :param n_classes:
        :param pretrained:
        :param none_classifier:
        :param init_weight_with_caffe: dict{method, weight_path}
        """
        super(FCN8s, self).__init__()
        self.n_classes = n_classes
        self.none_classifier = none_classifier

        vgg16 = models.vgg16(pretrained=pretrained)
        features = list(vgg16.features.children())

        if n_input_channel == 3 and pretrained:
            self.conv_block1 = nn.Sequential(*features[:5])

        else:
            self.conv_block1 = nn.Sequential(
                nn.Conv2d(n_input_channel, 64, 3, padding=100),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride=2, ceil_mode=True), )

        self.conv_block2 = nn.Sequential(*features[5:10])
        self.conv_block3 = nn.Sequential(*features[10:17])
        self.conv_block4 = nn.Sequential(*features[17:24])
        self.conv_block5 = nn.Sequential(*features[24:])

        if not self.none_classifier:
            self.classifier = nn.Sequential(
                nn.Conv2d(512, 4096, 7),
                nn.ReLU(inplace=True),
                nn.Dropout2d(),
                nn.Conv2d(4096, 4096, 1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(),
                nn.Conv2d(4096, self.n_classes, 1),
            )
        else:
            # fixme
            self.classifier = nn.Sequential(
                nn.Conv2d(512, self.n_classes, 1)
            )

        if init_weight_with_caffe is not None:
            _modules = []
            if init_weight_with_caffe['method'] == 'classifier':
                _modules = [self.classifier.__getitem__(0), self.classifier.__getitem__(3)]
            elif init_weight_with_caffe['method'] == 'seq':
                module_to_init = [self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4,
                                  self.conv_block5, self.classifier]
                for _module in module_to_init:
                    for m in _module.modules():
                        if isinstance(m, nn.Conv2d):
                            _modules.append(m)

            init_from_caffe(_modules, torch.load(init_weight_with_caffe['weight_path']))

        self.score_pool4 = nn.Conv2d(512, self.n_classes, 1)
        self.score_pool3 = nn.Conv2d(256, self.n_classes, 1)

    def conv_dict(self, x):
        conv1 = self.conv_block1(x)
        conv2 = self.conv_block2(conv1)
        conv3 = self.conv_block3(conv2)
        conv4 = self.conv_block4(conv3)
        conv5 = self.conv_block5(conv4)

        conv_list = [conv1, conv2, conv3, conv4, conv5]
        conv_dict = [('conv{}'.format(i + 1), m) for i, m in enumerate(conv_list)]
        conv_dict = dict(conv_dict)

        score = self.classifier(conv5)
        return score, conv_dict

    def up_score(self, score, conv_dict):
        """
        只有上採樣的過程不同
        :param score:
        :param conv_dict:
        :return:
        """
        score_pool4 = self.score_pool4(conv_dict['conv4'])
        score_pool3 = self.score_pool3(conv_dict['conv3'])

        score = F.upsample(score, size=score_pool4.size()[2:], mode='bilinear')
        score += score_pool4

        score = F.upsample(score, size=score_pool3.size()[2:], mode='bilinear')
        score += score_pool3
        return score

    def forward(self, x):
        score, conv_dict = self.conv_dict(x)
        score = self.up_score(score, conv_dict)
        out = F.upsample(score, x.size()[2:], mode='bilinear')
        return out
        #
        # def init_params(self, model, **kwargs):
        #     blocks = [self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4, self.conv_block5]
        #     layers = reduce(lambda x, y: x + y, map(lambda block: list(block.features.children()), blocks), [])
        #     features = list(model.features.children())
        #     for idx, src, des in enumerate(zip(features, layers)):
        #         if isinstance(src, nn.Conv2d) and isinstance(des, nn.Conv2d):
        #             assert src.weight.size() == des.weight.size()
        #             assert src.bias.size() == des.bias.size()
        #             des.weight.data = src.weight.data
        #             des.bias.data = src.bias.data
        #
        #     for idx_src, idx_des in zip([0, 3], [0, 3]):
        #         src = model.classifier[idx_src]
        #         des = model.classifier[idx_des]
        #         des.weight.data = src.weight.data.view(src.weight.data.size())
        #         des.bias.data = src.weight.data.view(src.bias.data.size())
        #
        # def init_from_caffe(self, caffe_info):
        #     """
        #     :param caffe_info: dict type {'params': dict, 'layers': list}, feature+classifier
        #     :return:
        #     """
        #     blocks = [self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4, self.conv_block5,
        #               self.classifier]
        #     modules = reduce(lambda x, y: x + y, map(lambda block: list(block.features.children()), blocks), [])
        #     conv_modules = list(filter(lambda x: isinstance(x, nn.Conv2d), modules))
        #
        #     params = caffe_info['params']
        #     layers = caffe_info['layers']
        #     assert len(layers) == len(conv_modules)
        #
        #     for idx, (lr, module) in enumerate(zip(layers, conv_modules)):
        #         assert lr.type == 'Convolution'
        #         module = conv_modules[idx]
        #         lr_params = params[lr.name]
        #         assert module.weight.data.size() == lr_params[0].data.shape and module.bias.data.size() == lr_params[
        #             1].data.shape
        #         module.weight.data = torch.from_numpy(lr_params[0].data)
        #         module.bias.data = torch.from_numpy(lr_params[1].data)
        #
        # def prepare_caffe(self, protofile, caffemodel):
        #     """
        #     調用以初始化
        #     :param protofile:
        #     :param caffemodel:
        #     :return:
        #     """
        #     rst_file = self.__class__.__name__ + '_caffe_weights.pkl'
        #     if not os.path.exists(rst_file):
        #         caffe_info = caffe2pytorch(protofile, caffemodel, rst_file, save_all=True)
        #     else:
        #         caffe_info = pickle.load(open(rst_file))
        #     self.init_from_caffe(caffe_info)


class FCN4s(FCN8s):
    def __init__(self, n_input_channel=3, n_classes=2):
        super(FCN4s, self).__init__(n_input_channel, n_classes)
        self.score_pool2 = nn.Conv2d(128, n_classes, 1)

    def up_score(self, score, conv_dict):
        score_pool4 = self.score_pool4(conv_dict['conv4'])
        score_pool3 = self.score_pool3(conv_dict['conv3'])
        score_pool2 = self.score_pool2(conv_dict['conv2'])

        score = F.upsample(score, size=score_pool4.size()[2:], mode='bilinear')
        score += score_pool4

        score = F.upsample(score, size=score_pool3.size()[2:], mode='bilinear')
        score += score_pool3

        score = F.upsample(score, size=score_pool2.size()[2:], mode='bilinear')
        score += score_pool2

        return score


class FCN2s(FCN8s):
    def __init__(self, n_input_channel=3,
                 n_classes=2,
                 pretrained=False,
                 none_classifier=False,
                 init_weight_with_caffe=None):
        """

        :param n_input_channel:
        :param n_classes:
        :param pretrained:
        :param none_classifier:
        :param init_weight_with_caffe:  dict{'method':'seq'||'classifier', 'weight_path':str}
        """
        super(FCN2s, self).__init__(n_input_channel,
                                    n_classes,
                                    pretrained=pretrained,
                                    none_classifier=none_classifier,
                                    init_weight_with_caffe=init_weight_with_caffe)
        self.score_pool2 = nn.Conv2d(128, n_classes, 1)
        self.score_pool1 = nn.Conv2d(64, n_classes, 1)

        ## init using xavier
        init_list = [self.score_pool4, self.score_pool3, self.score_pool2,
                     self.score_pool1]

        init_layers(init_xavier, *init_list)

    def up_score(self, score, conv_dict):
        score_pool4 = self.score_pool4(conv_dict['conv4'])
        score_pool3 = self.score_pool3(conv_dict['conv3'])
        score_pool2 = self.score_pool2(conv_dict['conv2'])
        score_pool1 = self.score_pool1(conv_dict['conv1'])

        score = F.upsample(score, size=score_pool4.size()[2:], mode='bilinear')
        score += score_pool4

        score = F.upsample(score, size=score_pool3.size()[2:], mode='bilinear')
        score += score_pool3

        score = F.upsample(score, size=score_pool2.size()[2:], mode='bilinear')
        score += score_pool2

        score = F.upsample(score, size=score_pool1.size()[2:], mode='bilinear')
        score += score_pool1

        return score

    def pretrained_parameters(self):
        params = []
        pretrained_layers = [self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4, self.conv_block5]
        for layer in pretrained_layers:
            params += list(layer.parameters())

        return params

    def none_pretrained_parameters(self):
        ignored_params = list(map(id, self.pretrained_parameters()))
        base_params = list(filter(lambda p: id(p) not in ignored_params, self.parameters()))
        return base_params


class FCN2sDeconv(nn.Module):

    def __init__(self, type='resnet152', none_classifier=False, pretrained=True, n_classes=2):
        super(FCN2sDeconv, self).__init__()

        self.n_classes = n_classes

        resnet = getattr(models, type)(pretrained=pretrained)

        self.conv_block1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )

        self.conv_block2 = nn.Sequential(
            resnet.maxpool,
            resnet.layer1
        )

        self.conv_block3 = resnet.layer2
        self.conv_block4 = resnet.layer3
        self.conv_block5 = resnet.layer4

        self.top_deconv = self._deconv(7, 1)
        self.deconv5 = self._deconv(4, 2)
        self.deconv4 = self._deconv(4, 2)
        self.deconv3 = self._deconv(4, 2)
        self.deconv2 = self._deconv(4, 2)
        self.deconv1 = self._deconv(4, 2)

        self.score_conv6 = nn.Sequential(
            nn.Conv2d(2048, 2048, 7, padding=0),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(2048, self.n_classes, 1),
        )

        self.score_conv5 = self._conv(2048)
        self.score_conv4 = self._conv(1024)
        self.score_conv3 = self._conv(512)
        self.score_conv2 = self._conv(256)
        self.score_conv1 = self._conv(64)

        init_layers(init_xavier, *[getattr(self, attr) for attr in dir(self) if 'block' not in str(attr)])

    def _conv(self, n_channels):
        return nn.Sequential(
            nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels, self.n_classes, 1),
        )

    def _deconv(self, ksize, stride):
        return nn.Sequential(
            nn.ConvTranspose2d(self.n_classes, self.n_classes, kernel_size=ksize, stride=stride),
        )

    def forward(self, x):
        out1 = self.conv_block1(x)
        out2 = self.conv_block2(out1)
        out3 = self.conv_block3(out2)
        out4 = self.conv_block4(out3)
        out5 = self.conv_block5(out4)

        top = self.score_conv6(out5)

        score5 = self.score_conv5(out5)
        score4 = self.score_conv4(out4)
        score3 = self.score_conv3(out3)
        score2 = self.score_conv2(out2)
        score1 = self.score_conv1(out1)

        upscore5 = self.top_deconv(top)
        upscore4 = self.deconv5(score5 + center_crop(upscore5, score5))
        upscore3 = self.deconv4(score4 + center_crop(upscore4, score4))
        upscore2 = self.deconv3(score3 + center_crop(upscore3, score3))
        upscore1 = self.deconv2(score2 + center_crop(upscore2, score2))
        upscore0 = self.deconv1(score1 + center_crop(upscore1, score1))
        out = center_crop(upscore0, x)
        return out


class FCN2sReset(FCN2s):
    """
    fcn2s + resnet
    :return:
    """

    resnet_modules = {
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152
    }

    model_loaded = None

    def __init__(self, n_input_channel=3, n_classes=2, type='resnet50', none_classifier=False):
        super(FCN2sReset, self).__init__()

        self.n_classes = n_classes
        self.n_input_channel = n_input_channel
        self.none_classifier = none_classifier

        resnet = getattr(models, type)(pretrained=True)

        self.conv_block1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )

        self.conv_block2 = nn.Sequential(
            resnet.maxpool,
            resnet.layer1
        )

        self.conv_block3 = resnet.layer2
        self.conv_block4 = resnet.layer3
        self.conv_block5 = resnet.layer4

        # fixme 这个地方的权重到底是不是要保留1、保留和vgg统一；2、不保留，resnet本来就没有这个结构
        if not self.none_classifier:
            self.classifier = nn.Sequential(
                nn.Conv2d(2048, 1024, 3, padding=3),  # change form 7 to 3
                nn.ReLU(inplace=True),
                nn.Dropout2d(),
                nn.Conv2d(1024, 1024, 1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(),
                nn.Conv2d(1024, self.n_classes, 1),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout2d(),
                nn.BatchNorm2d(2048),
                nn.ReLU(inplace=True),
                nn.Conv2d(2048, self.n_classes, 1),
                # nn.BatchNorm2d(1),
            )

        # bn+conv
        # fixme 在BN之后加入relu是不是更加的make sense ??
        self.score_pool4 = nn.Sequential(
            nn.BatchNorm2d(1024),
            # nn.ReLU(inplace=True), //原版的是removed(resnet152-fcn2s)
            nn.Conv2d(1024, self.n_classes, 1),
            # nn.BatchNorm2d(1),
        )
        self.score_pool3 = nn.Sequential(
            nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
            nn.Conv2d(512, self.n_classes, 1),
            # nn.BatchNorm2d(1),
        )
        self.score_pool2 = nn.Sequential(
            nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            nn.Conv2d(256, self.n_classes, 1),
            # nn.BatchNorm2d(1),
        )
        self.score_pool1 = nn.Sequential(
            nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            nn.Conv2d(64, self.n_classes, 1),
            # nn.BatchNorm2d(1),
        )
        ## init using xavier
        init_list = [self.score_pool4, self.score_pool3, self.score_pool2, self.score_pool1, self.classifier, ]
        init_layers(init_xavier, *init_list)

    def init_params(self, model, **kwargs):
        pass

    def pretrained_parameters(self):
        params = []
        pretrained_layers = [self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4, self.conv_block5]
        for layer in pretrained_layers:
            params += list(layer.parameters())

        return params

    def none_pretrained_parameters(self):
        ignored_params = list(map(id, self.pretrained_parameters()))
        base_params = list(filter(lambda p: id(p) not in ignored_params, self.parameters()))
        return base_params


#####################################################################
# prepare for refine part
####################################################################

class SRCNN(nn.Module):
    """
    超分辨率复原
    """

    def __init__(self, n_inputs, n_outputs, sigmoid_output=False, auto_pad=False):
        super(SRCNN, self).__init__()

        self.sigmoid_output = sigmoid_output
        self.auto_pad = auto_pad

        self.features = nn.Sequential(
            nn.Conv2d(n_inputs, 64, kernel_size=9, stride=1, padding=(4 if auto_pad else 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_outputs, kernel_size=5, stride=1, padding=(2 if auto_pad else 0))
        )

        init_list = list(self.features.children())
        init_layers(init_xavier, *init_list)

    def forward(self, x):
        x = self.features(x)
        if self.sigmoid_output:
            x = nn.Sigmoid(x)

        return x


class SRCNNResNetFCN2s(nn.Module):
    def __init__(self, weight_path=None):
        super(SRCNNResNetFCN2s, self).__init__()

        self.fcn2s = FCN2sReset(type='resnet152')
        # load weights
        if weight_path is not None:
            print("loading weights ...")
            state = torch.load(weight_path)['state_dict']
            self.fcn2s.load_state_dict(state)

        self.srcnn = SRCNN(n_inputs=5, auto_pad=True, n_outputs=2, sigmoid_output=False)

    def none_pretrained_parameters(self):
        return list(self.srcnn.parameters())

    def pretrained_parameters(self):
        return list(self.fcn2s.parameters())

    def forward(self, x):
        _out = self.fcn2s(x)
        x = torch.cat([_out, x], dim=1)

        x = self.srcnn(x)
        return x


#####################################################################


class FCN2sResnetFC(FCN2sReset):
    def __init__(self, n_input_channel=3, n_classes=2, type='resnet50'):
        super(FCN2sResnetFC, self).__init__(n_input_channel, n_classes, type)

        # resnet = FCN2sReset.model_loaded
        # self.avgpool = resnet.avgpool

        self.fc = nn.Linear(2048, n_classes)
        init_xavier(self.fc.weight)

    def forward(self, x):
        # 分割
        _score, conv_dict = self.conv_dict(x)
        score = self.up_score(_score, conv_dict)
        dense_predict = F.upsample(score, x.size()[2:], mode='bilinear')

        # 分类
        # x = self.avgpool(conv_dict['conv5'])
        conv = conv_dict['conv5']
        x = F.avg_pool2d(conv, conv.size(3), stride=1)  # pooling by auto size
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return dense_predict, x

    # todo test 这个对结果到底是不是有影响
    def none_pretrained_parameters(self):
        none_pretrained = super(FCN2sReset, self).none_pretrained_parameters()
        none_pretrained += list(self.fc.parameters())
        return none_pretrained


class FCN8sDenseNet(FCN8s):
    def __init__(self, n_input_channel=3, n_classes=2, densenet_type='densenet121', **kwargs):
        super(FCN8sDenseNet, self).__init__(n_input_channel, n_classes)
        densenet = self.get_densenet(densenet_type, **kwargs)
        feature_layers = dict(densenet.features.named_children())
        self.conv_block1 = nn.Sequential(
            feature_layers['conv0'],
            feature_layers['norm0'],
            feature_layers['relu0'],
        )

        self.conv_block2 = nn.Sequential(
            feature_layers['pool0'],
        )

        self.conv_block3 = nn.Sequential(
            feature_layers['denseblock1'],
            feature_layers['transition1']
        )

        self.conv_block4 = nn.Sequential(
            feature_layers['denseblock2'],
            feature_layers['transition2']
        )

        self.conv_block5 = nn.Sequential(
            feature_layers['denseblock3'],
            feature_layers['transition3']
        )

        self.classifier = nn.Sequential(
            feature_layers['denseblock4'],
            nn.Conv2d(1024, 1024, 3, padding=1),  # change kernel from 7 to 3
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(1024, 1024, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(1024, self.n_classes, 1),
        )

        self.score_pool4 = nn.Sequential(nn.BatchNorm2d(256), nn.Conv2d(256, self.n_classes, 1), )
        self.score_pool3 = nn.Sequential(nn.BatchNorm2d(128), nn.Conv2d(128, self.n_classes, 1), )

        ## init using xavier
        init_layers(init_xavier, self.classifier, self.score_pool4, self.score_pool3)

    def get_densenet(self, densenet, **kwargs):
        return getattr(models, densenet)(pretrained=kwargs['pretrained'])

    def init_params(self, model, **kwargs):
        pass


#####################################################################################################################
# 其他网络结构
#####################################################################################################################

def _vgg16_blocks(pretrained, with_bn):
    vgg = models.vgg16_bn(pretrained=pretrained) if with_bn else models.vgg16(pretrained=pretrained)
    if with_bn:
        layers = [(0, 6), (7, 6), (14, 9), (24, 9), (34, 9)]
    else:
        layers = [(0, 4), (5, 4), (10, 6), (17, 6), (24, 6)]

    features = list(vgg.features.children())
    conv1 = nn.Sequential(*features[layers[0][0]:layers[0][0] + layers[0][1]])  # 64
    conv2 = nn.Sequential(*features[layers[1][0]:layers[1][0] + layers[1][1]])  # 128
    conv3 = nn.Sequential(*features[layers[2][0]:layers[2][0] + layers[2][1]])  # 256
    conv4 = nn.Sequential(*features[layers[3][0]:layers[3][0] + layers[3][1]])  # 512
    conv5 = nn.Sequential(*features[layers[4][0]:layers[4][0] + layers[4][1]])  # 512
    return conv1, conv2, conv3, conv4, conv5


class FCN2sVGG(nn.Module):
    """
    FCN2s 使用VGG实现
    """

    def __init__(self, conv_list=None, pretrained=True, with_bn=True, n_classes=1):
        super(FCN2sVGG, self).__init__()

        self.n_classes = n_classes

        if conv_list is not None:
            self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4, self.conv_block5 = conv_list
        else:
            self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4, self.conv_block5 = _vgg16_blocks(
                pretrained, with_bn)

        self.deconv5 = self._deconv(4, 2)
        self.deconv4 = self._deconv(4, 2)
        self.deconv3 = self._deconv(4, 2)
        self.deconv2 = self._deconv(4, 2)
        self.deconv1 = self._deconv(4, 2)

        self.score_conv5 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(1024, self.n_classes, 1),
        )

        self.score_conv4 = self._conv(512)
        self.score_conv3 = self._conv(256)
        self.score_conv2 = self._conv(128)
        self.score_conv1 = self._conv(64)

        init_layers(init_xavier, *[getattr(self, attr) for attr in dir(self) if 'block' not in str(attr)])

    def _conv(self, n_channels):
        return nn.Sequential(
            nn.Conv2d(n_channels, self.n_classes, 1),
        )

    def _deconv(self, ksize, stride):
        return nn.Sequential(
            nn.ConvTranspose2d(self.n_classes, self.n_classes, kernel_size=ksize, stride=stride)
        )

    def forward(self, x):
        max_pool = partial(F.max_pool2d, kernel_size=2, stride=2)
        out1 = max_pool(self.conv_block1(x))
        out2 = max_pool(self.conv_block2(out1))
        out3 = max_pool(self.conv_block3(out2))
        out4 = max_pool(self.conv_block4(out3))
        out5 = max_pool(self.conv_block5(out4))

        score5 = self.score_conv5(out5)
        score4 = self.score_conv4(out4)
        score3 = self.score_conv3(out3)
        score2 = self.score_conv2(out2)
        score1 = self.score_conv1(out1)

        upscore4 = self.deconv5(score5)
        upscore3 = self.deconv4(score4 + center_crop(upscore4, score4))
        upscore2 = self.deconv3(score3 + center_crop(upscore3, score3))
        upscore1 = self.deconv2(score2 + center_crop(upscore2, score2))
        upscore0 = self.deconv1(score1 + center_crop(upscore1, score1))
        out = center_crop(upscore0, x)
        return out


class DeepHEDFCN2sBackResNet(nn.Module):
    """
    集合HED和FCN2sVGG 把HED作为SP放在FCN后面
    """

    def __init__(self, resnet_type='resnet50', pretrained=True, with_bn=False, n_classes=1):
        super(DeepHEDFCN2sBackResNet, self).__init__()

        self.n_classes = n_classes

        blocks = _resnet_blocks(resnet_type, pretrained)
        self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4, self.conv_block5 = blocks

        self._fcn()
        self._hed()

        init_layers(init_xavier, *[getattr(self, attr) for attr in dir(self) if 'block' not in str(attr)])
        self.pretrained_layers = [self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4,
                                  self.conv_block5]

    def pretrained_parameters(self):
        params = []
        for layer in self.pretrained_layers:
            params += list(layer.parameters())

        return params

    def none_pretrained_parameters(self):
        ignored_params = list(map(id, self.pretrained_parameters()))
        base_params = list(filter(lambda p: id(p) not in ignored_params, self.parameters()))
        return base_params

    def _fcn(self):
        def _conv(n_channels):
            return nn.Sequential(
                nn.BatchNorm2d(n_channels),
                nn.Conv2d(n_channels, self.n_classes, 1),
            )

        def _deconv(ksize, stride):
            return nn.Sequential(
                nn.ConvTranspose2d(self.n_classes, self.n_classes, kernel_size=ksize, stride=stride)
            )

        self.deconv5 = _deconv(4, 2)
        self.deconv4 = _deconv(4, 2)
        self.deconv3 = _deconv(4, 2)
        self.deconv2 = _deconv(4, 2)
        self.deconv1 = _deconv(4, 2)

        self.score_conv1 = _conv(64)
        self.score_conv2 = _conv(256)
        self.score_conv3 = _conv(512)
        self.score_conv4 = _conv(1024)
        self.score_conv5 = _conv(2048)

    def _hed(self):
        def _score_conv(n_inplanes, upsample=None):
            block = nn.Sequential(nn.Conv2d(n_inplanes, 1, kernel_size=1, stride=1), )  # TODO 增加一个卷积层用sobel算子初始化
            if upsample is not None:
                block.add_module('upsample', nn.ConvTranspose2d(1, 1, kernel_size=2 * upsample, stride=upsample))

            return block

        self.hed_score_conv1 = _score_conv(1, upsample=2)
        self.hed_score_conv2 = _score_conv(1, upsample=2 ** 2)
        self.hed_score_conv3 = _score_conv(1, upsample=2 ** 3)
        self.hed_score_conv4 = _score_conv(1, upsample=2 ** 4)
        self.hed_score_conv5 = _score_conv(1, upsample=2 ** 5)
        self.fuse = _score_conv(5)

    def forward(self, x):
        """
        :param x:
        :return:
            Segmentation None*1*h*w
            Edge detection: list with 6 ele when training 1 ele for testing same shape with segmentation
        """
        # fcn2s
        out1 = self.conv_block1(x)
        out2 = self.conv_block2(out1)
        out3 = self.conv_block3(out2)
        out4 = self.conv_block4(out3)
        out5 = self.conv_block5(out4)

        score5 = self.score_conv5(out5)
        score4 = self.score_conv4(out4)
        score3 = self.score_conv3(out3)
        score2 = self.score_conv2(out2)
        score1 = self.score_conv1(out1)

        upscore4 = self.deconv5(score5)
        upscore3 = self.deconv4(score4 + center_crop(upscore4, score4))
        upscore2 = self.deconv3(score3 + center_crop(upscore3, score3))
        upscore1 = self.deconv2(score2 + center_crop(upscore2, score2))
        upscore0 = self.deconv1(score1 + center_crop(upscore1, score1))
        seg_out = center_crop(upscore0, x)

        # hed
        hed_score5 = self.hed_score_conv5(score5)
        hed_score4 = self.hed_score_conv4(upscore4)
        hed_score3 = self.hed_score_conv3(upscore3)
        hed_score2 = self.hed_score_conv2(upscore2)
        hed_score1 = self.hed_score_conv1(upscore1)

        # print(hed_score5.size(), hed_score4.size(), hed_score3.size(), hed_score2.size(), hed_score1.size())

        cropped_score = list_center_crop([hed_score5, hed_score4, hed_score3, hed_score2, hed_score1], x)
        edge_fuse = self.fuse(torch.cat(cropped_score, dim=1))

        # result
        if self.training:
            cropped_score.insert(0, edge_fuse)
            return seg_out, cropped_score

        return seg_out, edge_fuse  # testing


class SobelLayer(nn.Module):
    kernel_w = np.asarray([[-1.0, 0.0, 1.0], [-2., 0., 2.], [-1., 0., 1.]], dtype=np.float32).reshape((1, 1, 3, 3))
    kernel_h = np.asarray([[1.0, 2.0, 1.0], [0.0, 0., 0.], [-1., -2., -1.]], dtype=np.float32).reshape((1, 1, 3, 3))

    def __init__(self, n_inputs):
        super(SobelLayer, self).__init__()
        self.conv_w = nn.Conv2d(n_inputs, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_h = nn.Conv2d(n_inputs, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv_w.weight.data = torch.from_numpy(np.repeat(SobelLayer.kernel_w, n_inputs, axis=0))
        self.conv_h.weight.data = torch.from_numpy(np.repeat(SobelLayer.kernel_h, n_inputs, axis=0))

    def forward(self, x):
        wx = self.conv_w(x)
        hx = self.conv_h(x)

        h = torch.sqrt(torch.pow(wx, 2) + torch.pow(hx, 2))
        return h


class DialationLayer(nn.Module):
    def __init__(self, n_inputs):
        super(DialationLayer, self).__init__()

    def forward(self, x):
        pass


class DeepHEDFCN2s(nn.Module):
    """
    集合HED和FCN2sVGG
    """

    def __init__(self, pretrained=True, with_bn=False, n_classes=1):
        super(DeepHEDFCN2s, self).__init__()

        self.n_classes = n_classes

        blocks = _vgg16_blocks(pretrained, with_bn)
        self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4, self.conv_block5 = blocks

        self._fcn()
        self._hed()

        init_layers(init_xavier, *[getattr(self, attr) for attr in dir(self) if 'block' not in str(attr)])

    def pretrained_parameters(self):
        params = []
        pretrained_layers = [self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4, self.conv_block5]
        for layer in pretrained_layers:
            params += list(layer.parameters())

        return params

    def none_pretrained_parameters(self):
        ignored_params = list(map(id, self.pretrained_parameters()))
        base_params = list(filter(lambda p: id(p) not in ignored_params, self.parameters()))
        return base_params

    def _fcn(self):
        def _conv(n_channels):
            return nn.Sequential(
                nn.Conv2d(n_channels, self.n_classes, 1),
                # SobelLayer(self.n_classes),
                # nn.Conv2d(self.n_classes, self.n_classes, 3),
            )

        def _deconv(ksize, stride):
            return nn.Sequential(
                nn.ConvTranspose2d(self.n_classes, self.n_classes, kernel_size=ksize, stride=stride)
            )

        self.deconv5 = _deconv(4, 2)
        self.deconv4 = _deconv(4, 2)
        self.deconv3 = _deconv(4, 2)
        self.deconv2 = _deconv(4, 2)
        self.deconv1 = _deconv(4, 2)

        self.score_conv5 = _conv(512)
        self.score_conv4 = _conv(512)
        self.score_conv3 = _conv(256)
        self.score_conv2 = _conv(128)
        self.score_conv1 = _conv(64)

    def _hed(self):

        def _score_conv(n_inplanes, upsample=None):
            block = nn.Sequential(nn.Conv2d(n_inplanes, 1, kernel_size=1, stride=1), )
            if upsample is not None:
                block.add_module('upsample', nn.ConvTranspose2d(1, 1, kernel_size=2 * upsample, stride=upsample))

            return block

        self.hed_score_conv1 = _score_conv(64)
        self.hed_score_conv2 = _score_conv(64 * 2, upsample=2 ** 1)
        self.hed_score_conv3 = _score_conv(64 * 4, upsample=2 ** 2)
        self.hed_score_conv4 = _score_conv(64 * 8, upsample=2 ** 3)
        self.hed_score_conv5 = _score_conv(64 * 8, upsample=2 ** 4)
        self.fuse = _score_conv(5)

    def forward(self, x):
        """
        :param x:
        :return:
            Segmentation None*1*h*w
            Edge detection: list with 6 ele when training 1 ele for testing same shape with segmentation
        """
        # fcn2s
        max_pool = partial(F.max_pool2d, kernel_size=2, stride=2)
        out1 = self.conv_block1(x)
        pool1 = max_pool(out1)
        out2 = self.conv_block2(pool1)
        pool2 = max_pool(out2)
        out3 = self.conv_block3(pool2)
        pool3 = max_pool(out3)
        out4 = self.conv_block4(pool3)
        pool4 = max_pool(out4)
        out5 = self.conv_block5(pool4)
        pool5 = max_pool(out5)

        score5 = self.score_conv5(pool5)
        score4 = self.score_conv4(pool4)
        score3 = self.score_conv3(pool3)
        score2 = self.score_conv2(pool2)
        score1 = self.score_conv1(pool1)

        upscore4 = self.deconv5(score5)
        upscore3 = self.deconv4(score4 + center_crop(upscore4, score4))
        upscore2 = self.deconv3(score3 + center_crop(upscore3, score3))
        upscore1 = self.deconv2(score2 + center_crop(upscore2, score2))
        upscore0 = self.deconv1(score1 + center_crop(upscore1, score1))
        seg_out = center_crop(upscore0, x)

        # hed
        hed_score5 = self.hed_score_conv5(out5)
        hed_score4 = self.hed_score_conv4(out4)
        hed_score3 = self.hed_score_conv3(out3)
        hed_score2 = self.hed_score_conv2(out2)
        hed_score1 = self.hed_score_conv1(out1)

        # print(hed_score5.size(), hed_score4.size(), hed_score3.size(), hed_score2.size(), hed_score1.size())

        cropped_score = list_center_crop([hed_score5, hed_score4, hed_score3, hed_score2, hed_score1], x)
        edge_fuse = self.fuse(torch.cat(cropped_score, dim=1))

        # result
        if self.training:
            cropped_score.insert(0, edge_fuse)
            return seg_out, cropped_score

        return seg_out, edge_fuse  # testing


def _resnet_blocks(resnet_type, pretrained):
    """
    :param resnet_type:
    :param pretrained:
    :return:
    """
    resnet = getattr(models, resnet_type)(pretrained=pretrained)
    conv_block1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 4
    conv_block2 = nn.Sequential(resnet.maxpool, resnet.layer1)
    conv_block3 = resnet.layer2  # 8
    conv_block4 = resnet.layer3  # 16
    conv_block5 = resnet.layer4  # 32

    return conv_block1, conv_block2, conv_block3, conv_block4, conv_block5


class EdgeHEDResNet(nn.Module):
    """
    Edge HED ResNet
    """

    def __init__(self, resnet_type='resnet101', pretrained=True):
        super(EdgeHEDResNet, self).__init__()
        self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4, self.conv_block5 = _resnet_blocks(
            resnet_type, pretrained)

        self._build()

        init_layers(init_xavier, *[getattr(self, attr) for attr in dir(self) if 'block' not in str(attr)])

    def _build(self):
        self.score_conv1 = self._score_conv(64, upsample=2 ** 1)
        self.score_conv2 = self._score_conv(256, upsample=2 ** 2)
        self.score_conv3 = self._score_conv(512, upsample=2 ** 3)
        self.score_conv4 = self._score_conv(1024, upsample=2 ** 4)
        self.score_conv5 = self._score_conv(2048, upsample=2 ** 5)

        self.fuse = self._score_conv(5)

    def _score_conv(self, n_inplanes, upsample=None):
        block = nn.Sequential(
            # nn.BatchNorm2d(n_inplanes),
            nn.Conv2d(n_inplanes, 1, kernel_size=1, stride=1))
        if upsample is not None:
            block.add_module('upsample', nn.ConvTranspose2d(1, 1, kernel_size=2 * upsample, stride=upsample))

        return block

    def pretrained_parameters(self):
        params = []
        pretrained_layers = [self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4, self.conv_block5]
        for layer in pretrained_layers:
            params += list(layer.parameters())

        return params

    def none_pretrained_parameters(self):
        ignored_params = list(map(id, self.pretrained_parameters()))
        base_params = list(filter(lambda p: id(p) not in ignored_params, self.parameters()))
        return base_params

    def forward(self, x):
        out1 = self.conv_block1(x)
        out2 = self.conv_block2(out1)
        out3 = self.conv_block3(out2)
        out4 = self.conv_block4(out3)
        out5 = self.conv_block5(out4)

        score5 = self.score_conv5(out5)
        score4 = self.score_conv4(out4)
        score3 = self.score_conv3(out3)
        score2 = self.score_conv2(out2)
        score1 = self.score_conv1(out1)

        # print(score1.size(), score2.size(), score3.size(), score4.size(), score5.size())

        cropped_score = list_center_crop([score5, score4, score3, score2, score1], x)

        fuse = self.fuse(torch.cat(cropped_score, dim=1))

        if self.training:
            cropped_score.insert(0, fuse)
            return cropped_score

        return fuse  # testing


class EdgeHED(nn.Module):
    """
    Edge Detection for HED
    """

    def __init__(self, conv_list=None, pretrained=True, with_bn=False):
        super(EdgeHED, self).__init__()
        if conv_list is not None:
            self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4, self.conv_block5 = conv_list
        else:
            self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4, self.conv_block5 = _vgg16_blocks(
                pretrained, with_bn)

        self._build()

    def _build(self):
        self.score_conv1 = self._score_conv(64)
        self.score_conv2 = self._score_conv(64 * 2, upsample=2)
        self.score_conv3 = self._score_conv(64 * 4, upsample=2 ** 2)
        self.score_conv4 = self._score_conv(64 * 8, upsample=2 ** 3)
        self.score_conv5 = self._score_conv(64 * 8, upsample=2 ** 4)

        self.fuse = self._score_conv(5)

        init_layers(init_xavier, *[getattr(self, attr) for attr in dir(self) if 'block' not in str(attr)])

    def _score_conv(self, n_inplanes, upsample=None):
        block = nn.Sequential(nn.Conv2d(n_inplanes, 1, kernel_size=1, stride=1), )
        if upsample is not None:
            block.add_module('upsample', nn.ConvTranspose2d(1, 1, kernel_size=2 * upsample, stride=upsample))

        return block

    def forward(self, x):
        out1 = self.conv_block1(x)
        out2 = self.conv_block2(F.max_pool2d(out1, kernel_size=2, stride=2))
        out3 = self.conv_block3(F.max_pool2d(out2, kernel_size=2, stride=2))
        out4 = self.conv_block4(F.max_pool2d(out3, kernel_size=2, stride=2))
        out5 = self.conv_block5(F.max_pool2d(out4, kernel_size=2, stride=2))

        score5 = self.score_conv5(out5)
        score4 = self.score_conv4(out4)
        score3 = self.score_conv3(out3)
        score2 = self.score_conv2(out2)
        score1 = self.score_conv1(out1)

        # print(score1.size(), score2.size(), score3.size(), score4.size(), score5.size())

        cropped_score = list_center_crop([score5, score4, score3, score2, score1], x)

        fuse = self.fuse(torch.cat(cropped_score, dim=1))

        if self.training:
            cropped_score.insert(0, fuse)
            return cropped_score

        return fuse  # testing


class EdgeRCF(nn.Module):
    """
    Edge Detection RCF
    """

    def __init__(self, pretrained=True, with_bn=False):
        super(EdgeRCF, self).__init__()


class DeepSupervisedFCN(nn.Module):
    """
    DeepSupervisedFCN salience detection
    """

    def __init__(self, num_classes=1, with_bn=True, pretrained=True):
        """
        :param with_bn:
        :param mode: train test
        """
        super(DeepSupervisedFCN, self).__init__()

        self.num_classes = num_classes
        self.conv1, self.conv2, self.conv3, self.conv4, self.conv5 = _vgg16_blocks(pretrained, with_bn)

        self.pool5a = nn.AvgPool2d(kernel_size=3, padding=1, stride=1)

        self.dsn6 = self.__conv_block(7, 512, 512, 6, with_bn=with_bn)
        self.dsn5 = self.__conv_block(5, 512, 512, 5, with_bn=with_bn)
        self.dsn4 = self.__conv_block(5, 512, 256, 4, with_bn=with_bn)
        self.dsn3 = self.__conv_block(5, 256, 256, 3, with_bn=with_bn)
        self.dsn2 = self.__conv_block(3, 128, 128, 2, with_bn=with_bn)
        self.dsn1 = self.__conv_block(3, 64, 128, 1, with_bn=with_bn)

        self.upsample32_in_dsn6 = nn.ConvTranspose2d(num_classes, num_classes, 64, stride=32, bias=False)
        self.upsample32_dsn6 = nn.ConvTranspose2d(num_classes, num_classes, 64, stride=32, bias=False)
        self.upsample16_dsn6 = nn.ConvTranspose2d(num_classes, num_classes, 32, stride=16, bias=False)
        self.upsample8_dsn6 = nn.ConvTranspose2d(num_classes, num_classes, 16, stride=8, bias=False)
        self.upsample4_dsn6 = nn.ConvTranspose2d(num_classes, num_classes, 8, stride=4, bias=False)

        self.upsample16_in_dsn5 = nn.ConvTranspose2d(num_classes, num_classes, 32, stride=16, bias=False)
        self.upsample16_dsn5 = nn.ConvTranspose2d(num_classes, num_classes, 32, stride=16, bias=False)
        self.upsample8_dsn5 = nn.ConvTranspose2d(num_classes, num_classes, 16, stride=8, bias=False)
        self.upsample4_dsn5 = nn.ConvTranspose2d(num_classes, num_classes, 8, stride=4, bias=False)
        self.upsample2_dsn5 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, bias=False)

        self.upsample8_in_dsn4 = nn.ConvTranspose2d(num_classes, num_classes, 16, stride=8, bias=False)  # conv4-dsn4
        self.upsample8_dsn4 = nn.ConvTranspose2d(num_classes, num_classes, 16, stride=8, bias=False)
        self.upsample4_dsn4 = nn.ConvTranspose2d(num_classes, num_classes, 8, stride=4, bias=False)

        self.upsample4_in_dsn3 = nn.ConvTranspose2d(num_classes, num_classes, 8, stride=4, bias=False)  # conv4-dsn4
        self.upsample4_dsn3 = nn.ConvTranspose2d(num_classes, num_classes, 8, stride=4, bias=False)
        self.upsample2_dsn3 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, bias=False)

        self.upsample2_in_dsn2 = nn.ConvTranspose2d(num_classes, num_classes, 4, stride=2, bias=False)  # conv4-dns2

        self.conv4_dsn4 = nn.Conv2d(3 * num_classes, num_classes, kernel_size=1, stride=1)
        self.conv4_dsn3 = nn.Conv2d(3 * num_classes, num_classes, kernel_size=1, stride=1)
        self.conv4_dsn2 = nn.Conv2d(5 * num_classes, num_classes, kernel_size=1, stride=1)
        self.conv4_dsn1 = nn.Conv2d(5 * num_classes, num_classes, kernel_size=1, stride=1)

        self.fuse = nn.Conv2d(6 * num_classes, num_classes, kernel_size=1, stride=1)  # up 1/2/3/4/5/6

        init_layers(init_xavier, *[getattr(self, attr) for attr in dir(self) if 'dsn' in str(attr)])

    def pretrained_parameters(self):
        params = []
        pretrained_layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]
        for layer in pretrained_layers:
            params += list(layer.parameters())

        return params

    def none_pretrained_parameters(self):
        ignored_params = list(map(id, self.pretrained_parameters()))
        base_params = list(filter(lambda p: id(p) not in ignored_params, self.parameters()))
        return base_params

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(F.max_pool2d(x1, kernel_size=2, stride=2))
        x3 = self.conv3(F.max_pool2d(x2, kernel_size=2, stride=2))
        x4 = self.conv4(F.max_pool2d(x3, kernel_size=2, stride=2))
        x5 = self.conv5(F.max_pool2d(x4, kernel_size=2, stride=2))
        xp5a = self.pool5a(F.max_pool2d(x5, kernel_size=2, stride=2))

        dsn6 = self.dsn6(xp5a)
        dsn5 = self.dsn5(x5)
        dsn4 = self.dsn4(x4)
        dsn3 = self.dsn3(x3)
        dsn2 = self.dsn2(x2)
        dsn1 = self.dsn1(x1)

        up32_in_dns6 = self.upsample32_in_dsn6(dsn6)
        up32_dsn6 = self.upsample32_dsn6(dsn6)
        up16_dsn6 = self.upsample16_dsn6(dsn6)
        up8_dsn6 = self.upsample8_dsn6(dsn6)
        up4_dsn6 = self.upsample4_dsn6(dsn6)

        up16_in_dsn5 = self.upsample16_in_dsn5(dsn5)
        up16_dsn5 = self.upsample16_dsn5(dsn5)
        up8_dsn5 = self.upsample8_dsn5(dsn5)
        up4_dsn5 = self.upsample4_dsn5(dsn5)
        up2_dsn5 = self.upsample2_dsn5(dsn5)

        up8_dsn4 = self.upsample8_dsn4(dsn4)
        up4_dsn4 = self.upsample4_dsn4(dsn4)

        up4_dsn3 = self.upsample4_dsn3(dsn3)
        up2_dsn3 = self.upsample2_dsn3(dsn3)

        # concat dsn4 : conv3-dsn4 dsn6/5-4
        cat_4_6, cat_4_5 = list_center_crop([up4_dsn6, up2_dsn5], dsn4)
        cat_4 = torch.cat([dsn4, cat_4_6, cat_4_5], dim=1)
        conv4_dsn4 = self.conv4_dsn4(cat_4)
        up8_in_dsn4 = self.upsample8_in_dsn4(conv4_dsn4)
        # concat dsn3 : conv3-dsn3 dsn6/5-3
        cat_3_6, cat_3_5 = list_center_crop([up8_dsn6, up4_dsn5], dsn3)
        cat_3 = torch.cat([dsn3, cat_3_6, cat_3_5], dim=1)
        conv4_dsn3 = self.conv4_dsn3(cat_3)
        up4_in_dsn3 = self.upsample4_in_dsn3(conv4_dsn3)
        # concat dsn2 : conv3-dsn2 dsn6/5/4/3-2
        cat_2_6, cat_2_5, cat_2_4, cat_2_3 = list_center_crop([up16_dsn6, up8_dsn5, up4_dsn4, up2_dsn3], dsn2)
        cat_2 = torch.cat([dsn2, cat_2_6, cat_2_5, cat_2_4, cat_2_3], dim=1)
        conv4_dsn2 = self.conv4_dsn2(cat_2)
        up2_in_dsn2 = self.upsample2_in_dsn2(conv4_dsn2)
        # concat dsn1 : conv3-dsn1 dsn6/5/4/3-1
        cat_1_6, cat_1_5, cat_1_4, cat_1_3 = list_center_crop([up32_dsn6, up16_dsn5, up8_dsn4, up4_dsn3], dsn1)
        cat_1 = torch.cat([dsn1, cat_1_6, cat_1_5, cat_1_4, cat_1_3], dim=1)
        conv4_dsn1 = self.conv4_dsn1(cat_1)

        cropped = list_center_crop([conv4_dsn1, up2_in_dsn2, up4_in_dsn3, up8_in_dsn4, up16_in_dsn5, up32_in_dns6],
                                   target=x)
        # fuse
        fuse = self.fuse(torch.cat(cropped, dim=1))

        if self.training:
            cropped.append(fuse)
            return cropped
        else:
            return fuse

    def __conv_block(self, kernel_size, in_channels, first_out_channels, idx, with_bn=False):
        layer = nn.Sequential()
        layer.add_module('conv1-dns{}'.format(idx), nn.Conv2d(in_channels, first_out_channels,
                                                              kernel_size=kernel_size, padding=kernel_size // 2))
        if with_bn:
            layer.add_module('bn1-dns{}'.format(idx), nn.BatchNorm2d(first_out_channels))

        layer.add_module('relu1-dns{}'.format(idx), nn.ReLU(inplace=True))

        layer.add_module('conv2-dns{}'.format(idx),
                         nn.Conv2d(first_out_channels, first_out_channels,
                                   kernel_size=kernel_size, padding=kernel_size // 2))
        if with_bn:
            layer.add_module('bn2-dns{}'.format(idx), nn.BatchNorm2d(first_out_channels))

        layer.add_module('relu2-dns{}'.format(idx), nn.ReLU(inplace=True))

        layer.add_module('conv3-dns{}'.format(idx),
                         nn.Conv2d(in_channels=first_out_channels, out_channels=self.num_classes, kernel_size=1,
                                   padding=1))

        return layer


class SegNet(nn.Module):
    def __init__(self, input_channels=3, begin_channels=64, pretrained=False, with_bn=False):
        super(SegNet, self).__init__()

        self.with_bn = with_bn
        self.pretrained = pretrained

        if begin_channels == 64 and input_channels == 3:
            self.conv1, self.conv2, self.conv3, self.conv4, self.conv5 = _vgg16_blocks(pretrained, with_bn)

        self.deconv5 = self._deconv_block(512, 512, 3)
        self.deconv4 = self._deconv_block(512, 256, 3)
        self.deconv3 = self._deconv_block(256, 128, 3)
        self.deconv2 = self._deconv_block(128, 64, 2)
        self.deconv1 = self._deconv_block(64, 2, 2, without_last=True)

        init_layers(init_xavier, *[getattr(self, 'deconv{}'.format(i + 1)) for i in range(5)])

    def pretrained_parameters(self):
        if not self.pretrained:
            return []
        else:
            pms = []
            for i in range(5):
                pms += list(getattr(self, 'conv{}'.format(i + 1)).parameters())
            return pms

    def none_pretrained_parameters(self):
        pms = []
        for i in range(5):
            pms += list(getattr(self, 'deconv{}'.format(i + 1)).parameters())
        return pms

    def _deconv_block(self, input_channels, output_channels, n_conv, without_last=False):
        block = nn.Sequential()
        for idx in range(n_conv):
            conv_output_channels = input_channels if idx + 1 != n_conv else output_channels
            conv_layer = nn.Conv2d(input_channels, conv_output_channels, kernel_size=3, stride=1, padding=1)
            block.add_module('conv{}'.format(idx), conv_layer)

            if without_last and idx + 1 == n_conv:
                break

            if self.with_bn:
                block.add_module('bn{}'.format(idx), nn.BatchNorm2d(conv_output_channels))
            block.add_module('relu{}'.format(idx), nn.ReLU(inplace=True))

        return block

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_mp1, indices1 = F.max_pool2d(out_conv1, kernel_size=2, stride=2, return_indices=True)
        out_conv2 = self.conv2(out_mp1)
        out_mp2, indices2 = F.max_pool2d(out_conv2, kernel_size=2, stride=2, return_indices=True)
        out_conv3 = self.conv3(out_mp2)
        out_mp3, indices3 = F.max_pool2d(out_conv3, kernel_size=2, stride=2, return_indices=True)
        out_conv4 = self.conv4(out_mp3)
        out_mp4, indices4 = F.max_pool2d(out_conv4, kernel_size=2, stride=2, return_indices=True)
        out_conv5 = self.conv5(out_mp4)
        out_mp5, indices5 = F.max_pool2d(out_conv5, kernel_size=2, stride=2, return_indices=True)

        x = F.max_unpool2d(out_mp5, indices5, stride=2, kernel_size=2, output_size=out_conv5.size()[-2:])
        x = self.deconv5(x)
        x = F.max_unpool2d(x, indices4, stride=2, kernel_size=2, output_size=out_conv4.size()[-2:])
        x = self.deconv4(x)
        x = F.max_unpool2d(x, indices3, stride=2, kernel_size=2, output_size=out_conv3.size()[-2:])
        x = self.deconv3(x)
        x = F.max_unpool2d(x, indices2, stride=2, kernel_size=2, output_size=out_conv2.size()[-2:])
        x = self.deconv2(x)
        x = F.max_unpool2d(x, indices1, stride=2, kernel_size=2, output_size=out_conv1.size()[-2:])
        x = self.deconv1(x)
        return x


###########################################

class _PyramidPoolingModule(nn.Module):
    def __init__(self, in_dim, reduction_dim, setting):
        super(_PyramidPoolingModule, self).__init__()
        self.features = []
        for s in setting:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim, momentum=.95),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.upsample(f(x), x_size[2:], mode='bilinear'))
        out = torch.cat(out, 1)
        return out


class PSPNet(nn.Module):
    def __init__(self, num_classes, model_type="resnet101", pretrained=True, use_aux=True):
        super(PSPNet, self).__init__()
        self.use_aux = use_aux
        resnet = getattr(models, model_type)(pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        self.ppm = _PyramidPoolingModule(2048, 512, (1, 2, 3, 6))
        self.final = nn.Sequential(
            nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

        if use_aux:
            self.aux_logits = nn.Conv2d(1024, num_classes, kernel_size=1)
            init_layers(init_xavier, self.aux_logits)

        init_layers(init_xavier, self.ppm, self.final)

    def forward(self, x):
        x_size = x.size()
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.training and self.use_aux:
            aux = self.aux_logits(x)
        x = self.layer4(x)
        x = self.ppm(x)
        x = self.final(x)
        if self.training and self.use_aux:
            return F.upsample(x, x_size[2:], mode='bilinear'), F.upsample(aux, x_size[2:], mode='bilinear')
        return F.upsample(x, x_size[2:], mode='bilinear')


###########################################

class UNetV2(nn.Module):
    """
    根据经典额网络来的，高层的语意信息对分割不一定好，可以适当的降低U型网络的深度
    """

    def __init__(self, input_channels=3, begin_channels=64, expansion=2, with_bn=True, n_classes=2, pretrained=False):
        """

        :param input_channels: 网络的输入通道
        :param begin_channels: 初始特征通道数
        :param expansion: 乘子
        :param with_bn: 归一化
        :param n_classes: n的类别
        :param pretrained: 预训练模型
        """
        super(UNetV2, self).__init__()

        self.with_bn = with_bn
        self.pretrained = pretrained

        if begin_channels == 64 and input_channels == 3 and pretrained:
            vgg = models.vgg16_bn(pretrained=True) if with_bn else models.vgg16(pretrained=True)
            if with_bn:
                layers = [(0, 6), (7, 6), (14, 9), (24, 9), (34, 9)]
            else:
                layers = [(0, 4), (5, 4), (10, 6), (17, 6), (24, 6)]
            features = list(vgg.features.children())
            self.conv1, self.conv2, self.conv3, self.conv4, self.conv5 = _vgg16_blocks(pretrained, with_bn)

            to_copy = features[layers[4][0]:(layers[4][0] + layers[4][1] * 2 // 3)]
            import copy
            copyed = copy.deepcopy(to_copy)
            self.conv6 = nn.Sequential(*copyed)
        else:
            self.conv1 = self._conv_block(input_channels, begin_channels, n_conv=2)
            self.conv2 = self._conv_block(begin_channels, begin_channels * expansion, n_conv=2)
            self.conv3 = self._conv_block(begin_channels * (expansion ** 1), begin_channels * (expansion ** 2),
                                          n_conv=3)
            self.conv4 = self._conv_block(begin_channels * (expansion ** 2), begin_channels * (expansion ** 3),
                                          n_conv=3)
            self.conv5 = self._conv_block(begin_channels * (expansion ** 3), begin_channels * (expansion ** 3),
                                          n_conv=3)
            self.conv6 = self._conv_block(begin_channels * (expansion ** 3), begin_channels * (expansion ** 3),
                                          n_conv=2)

        self.deconv5 = self._conv_block(begin_channels * (expansion ** 4), begin_channels * (expansion ** 3), n_conv=2)
        self.deconv4 = self._conv_block(begin_channels * (expansion ** 4), begin_channels * (expansion ** 3), n_conv=2)
        self.deconv3 = self._conv_block(begin_channels * (expansion ** 3), begin_channels * (expansion ** 2), n_conv=2)
        self.deconv2 = self._conv_block(begin_channels * (expansion ** 2), begin_channels * (expansion ** 1), n_conv=2)
        self.deconv1 = self._conv_block(begin_channels * (expansion ** 1), begin_channels, n_conv=2)

        self.score = nn.Conv2d(begin_channels, n_classes, kernel_size=3, padding=1)

        self.upsample6 = self._deconv(begin_channels * (expansion ** 3), begin_channels * (expansion ** 3))
        self.upsample5 = self._deconv(begin_channels * (expansion ** 3), begin_channels * (expansion ** 3))
        self.upsample4 = self._deconv(begin_channels * (expansion ** 3), begin_channels * (expansion ** 2))
        self.upsample3 = self._deconv(begin_channels * (expansion ** 2), begin_channels * (expansion ** 1))
        self.upsample2 = self._deconv(begin_channels * (expansion ** 1), begin_channels)

        if not pretrained:
            init_layers(init_xavier, *[getattr(self, 'conv{}'.format(i + 1)) for i in range(6)])

        init_layers(init_xavier, *[getattr(self, 'deconv{}'.format(i + 1)) for i in range(5)])
        init_layers(init_xavier, *[getattr(self, 'upsample{}'.format(6 - i)) for i in range(5)])
        init_xavier(self.score.weight, self.score.bias)
        init_layers(init_xavier, self.conv6)

    def pretrained_parameters(self):
        if not self.pretrained:
            return []
        else:
            pms = []
            for i in range(4):
                pms += list(getattr(self, 'conv{}'.format(i + 1)).parameters())
            return pms

    def none_pretrained_parameters(self):
        ignored_params = list(map(id, self.pretrained_parameters()))
        base_params = list(filter(lambda p: id(p) not in ignored_params,
                                  self.parameters()))
        return base_params

    def _deconv(self, in_cahnnel, out_channel):
        return nn.Sequential(
            nn.ConvTranspose2d(in_cahnnel, out_channel, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

    def _conv_block(self, in_channel, out_channel, n_conv=2, ksize=3, stride=1, padding=1):
        block = nn.Sequential()
        for idx in range(n_conv):
            conv_in_channels = in_channel if idx == 0 else out_channel
            conv_layer = nn.Conv2d(conv_in_channels, out_channel, kernel_size=ksize, stride=stride, padding=padding)
            block.add_module('conv{}'.format(idx), conv_layer)
            if self.with_bn:
                block.add_module('bn{}'.format(idx), nn.BatchNorm2d(out_channel))
            block.add_module('relu{}'.format(idx), nn.ReLU(inplace=True))

        return block

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(F.max_pool2d(out_conv1, kernel_size=2, stride=2))
        out_conv3 = self.conv3(F.max_pool2d(out_conv2, kernel_size=2, stride=2))
        out_conv4 = self.conv4(F.max_pool2d(out_conv3, kernel_size=2, stride=2))
        out_conv5 = self.conv5(F.max_pool2d(out_conv4, kernel_size=2, stride=2))

        out_conv6 = self.conv6(F.max_pool2d(out_conv5, kernel_size=2, stride=2))

        out_deconv5 = self.deconv5(torch.cat((self.upsample6(out_conv6), out_conv5), 1))
        out_deconv4 = self.deconv4(torch.cat((self.upsample5(out_deconv5), out_conv4), 1))
        out_deconv3 = self.deconv3(torch.cat((self.upsample4(out_deconv4), out_conv3), 1))
        out_deconv2 = self.deconv2(torch.cat((self.upsample3(out_deconv3), out_conv2), 1))
        out_deconv1 = self.deconv1(torch.cat((self.upsample2(out_deconv2), out_conv1), 1))

        # print(out_deconv5.size(), out_deconv4.size(), out_deconv3.size(), out_deconv2.size(), out_deconv1.size())

        score = self.score(out_deconv1)
        return score


class UNet(nn.Module):
    """
    根据经典额网络来的，高层的语意信息对分割不一定好，可以适当的降低U型网络的深度
    """

    def __init__(self, input_channels=3, begin_channels=64, expansion=2, with_bn=True, n_classes=2, pretrained=False):
        """

        :param input_channels: 网络的输入通道
        :param begin_channels: 初始特征通道数
        :param expansion: 乘子
        :param with_bn: 归一化
        :param n_classes: n的类别
        :param pretrained: 预训练模型
        """
        super(UNet, self).__init__()

        self.with_bn = with_bn
        self.pretrained = pretrained

        if begin_channels == 64 and input_channels == 3 and pretrained:
            vgg = models.vgg16_bn(pretrained=True) if with_bn else models.vgg16(pretrained=True)
            if with_bn:
                layers = [(0, 6), (7, 6), (14, 9), (24, 9), (34, 9)]
            else:
                layers = [(0, 4), (5, 4), (10, 6), (17, 6), (24, 6)]
            features = list(vgg.features.children())
            self.conv1 = nn.Sequential(*features[layers[0][0]:layers[0][0] + layers[0][1]])
            self.conv2 = nn.Sequential(*features[layers[1][0]:layers[1][0] + layers[1][1]])
            self.conv3 = nn.Sequential(*features[layers[2][0]:layers[2][0] + layers[2][1]])
            self.conv4 = nn.Sequential(*features[layers[3][0]:layers[3][0] + layers[3][1]])
        else:
            self.conv1 = self._conv_block(input_channels, begin_channels, n_conv=2)
            self.conv2 = self._conv_block(begin_channels, begin_channels * expansion, n_conv=2)
            self.conv3 = self._conv_block(begin_channels * (expansion ** 1), begin_channels * (expansion ** 2),
                                          n_conv=2)
            self.conv4 = self._conv_block(begin_channels * (expansion ** 2), begin_channels * (expansion ** 3),
                                          n_conv=2)

        self.conv5 = self._conv_block(begin_channels * (expansion ** 3), begin_channels * (expansion ** 4), n_conv=1)

        self.deconv5 = self._conv_block(begin_channels * (expansion ** 4), begin_channels * (expansion ** 4), n_conv=1)
        self.deconv4 = self._conv_block(begin_channels * (expansion ** 4), begin_channels * (expansion ** 3), n_conv=2)
        self.deconv3 = self._conv_block(begin_channels * (expansion ** 3), begin_channels * (expansion ** 2), n_conv=2)
        self.deconv2 = self._conv_block(begin_channels * (expansion ** 2), begin_channels * (expansion ** 1), n_conv=2)
        self.deconv1 = self._conv_block(begin_channels * (expansion ** 1), begin_channels, n_conv=2)

        self.score = nn.Conv2d(begin_channels, n_classes, kernel_size=3, padding=1)

        self.upsample5 = self._deconv(begin_channels * (expansion ** 4), begin_channels * (expansion ** 3))
        self.upsample4 = self._deconv(begin_channels * (expansion ** 3), begin_channels * (expansion ** 2))
        self.upsample3 = self._deconv(begin_channels * (expansion ** 2), begin_channels * (expansion ** 1))
        self.upsample2 = self._deconv(begin_channels * (expansion ** 1), begin_channels)

        if not pretrained:
            init_layers(init_xavier, *[getattr(self, 'conv{}'.format(i + 1)) for i in range(5)])

        init_layers(init_xavier, *[getattr(self, 'deconv{}'.format(i + 1)) for i in range(5)])
        init_layers(init_xavier, *[getattr(self, 'upsample{}'.format(5 - i)) for i in range(4)])
        init_xavier(self.score.weight, self.score.bias)
        init_layers(init_xavier, self.conv5)

    def pretrained_parameters(self):
        if not self.pretrained:
            return []
        else:
            pms = []
            for i in range(4):
                pms += list(getattr(self, 'conv{}'.format(i + 1)).parameters())
            return pms

    def none_pretrained_parameters(self):
        ignored_params = list(map(id, self.pretrained_parameters()))
        base_params = list(filter(lambda p: id(p) not in ignored_params,
                                  self.parameters()))
        return base_params

    def _deconv(self, in_cahnnel, out_channel):
        return nn.Sequential(
            nn.ConvTranspose2d(in_cahnnel, out_channel, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

    def _conv_block(self, in_channel, out_channel, n_conv=2, ksize=3, stride=1, padding=1):
        block = nn.Sequential()
        for idx in range(n_conv):
            conv_in_channels = in_channel if idx == 0 else out_channel
            conv_layer = nn.Conv2d(conv_in_channels, out_channel, kernel_size=ksize, stride=stride, padding=padding)
            block.add_module('conv{}'.format(idx), conv_layer)
            if self.with_bn:
                block.add_module('bn{}'.format(idx), nn.BatchNorm2d(out_channel))
            block.add_module('relu{}'.format(idx), nn.ReLU(inplace=True))

        return block

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(F.max_pool2d(out_conv1, kernel_size=2, stride=2))
        out_conv3 = self.conv3(F.max_pool2d(out_conv2, kernel_size=2, stride=2))
        out_conv4 = self.conv4(F.max_pool2d(out_conv3, kernel_size=2, stride=2))
        out_conv5 = self.conv5(F.max_pool2d(out_conv4, kernel_size=2, stride=2))

        out_deconv5 = self.deconv5(out_conv5)
        out_deconv4 = self.deconv4(torch.cat((self.upsample5(out_deconv5), out_conv4), 1))
        out_deconv3 = self.deconv3(torch.cat((self.upsample4(out_deconv4), out_conv3), 1))
        out_deconv2 = self.deconv2(torch.cat((self.upsample3(out_deconv3), out_conv2), 1))
        out_deconv1 = self.deconv1(torch.cat((self.upsample2(out_deconv2), out_conv1), 1))

        # print(out_deconv5.size(), out_deconv4.size(), out_deconv3.size(), out_deconv2.size(), out_deconv1.size())

        score = self.score(out_deconv1)
        return score


class UNetResNet(nn.Module):
    """
    TODO 实现
    """

    def __init__(self):
        super(UNetResNet, self).__init__()


class SegNetResNet(nn.Module):
    """没法实现，因为ResNet下采样不是通过pooling实现    """

    def __init__(self):
        super(SegNetResNet, self).__init__()


####### test###############
def pass_test_case():
    # fcn8s = FCN8sDenseNet(pretrained=False)
    # for name, module in fcn8s.named_children():
    #     print(name)

    fcn = FCN8s()
    for name, m in fcn.named_modules():
        if isinstance(m, nn.Conv2d):
            print(name, m.weight.data.shape, m.bias.data.shape)
            # m = torch.randn((10, 3, 224, 224))
            # model = UNet(pretrained=True, with_bn=False)
            # # print(model.parameters())
            # print(model.pretrained_parameters())
            # # print(model.none_pretrained_parameters())
            # # print(list(model.parameters()))
            # # model.cuda()
            # # print(model)
            # # out = model(Variable(m).cuda())
            # # print(out.size())


class SRCNNResNetFCN2sTest(unittest.TestCase):
    def setUp(self):
        weight_path = '/home/adam/Gits/blur-detection/models/resnet152-fcn2s/epoch-100.pth'
        self.model = SRCNNResNetFCN2s(weight_path=None)
        print(list(self.model.none_pretrained_parameters()))

    def tearDown(self):
        pass

    def testForward(self):
        data = torch.rand(4, 3, 256, 256)
        out = self.model(Variable(data))
        print(out.shape)
        print(out)


class DeepSupervisedFCNTest(unittest.TestCase):
    def setUp(self):
        self.model = DeepSupervisedFCN(with_bn=False, num_classes=2)
        self.model.cuda()
        print(self.model)
        from src.blur import loss
        self.loss_fn = loss.DeepSupervisedLoss(loss_fn=AL2Loss2d(num_classes=2))

    def tearDown(self):
        pass

    def testModel(self):
        data = torch.randn((4, 3, 256, 256))
        label = Variable(torch.LongTensor(4, 1, 256, 256).random_(2)).cuda()
        vdata = Variable(data).cuda()
        outputs = self.model(vdata)
        print(outputs[0].shape)
        loss = self.loss_fn(outputs, label)
        print(loss)


class UNetV2Test(unittest.TestCase):
    def setUp(self):
        # self.model = UNetV2(pretrained=True)
        # self.model = PSPNet(2, model_type='resnet50', pretrained=True)
        self.model = DeepHEDFCN2sBackResNet(pretrained=False)
        print(self.model)

    def testModel(self):
        data = Variable(torch.randn((4, 3, 224, 224)))
        label = Variable(torch.rand((4, 1, 224, 224)))
        outputs = self.model(data)
        # print(outputs)
        # print(extra.shape)


if __name__ == '__main__':
    # print('\n'.join((dir(DeepSupervisedFCN()))))
    pass
