# coding: utf-8
from __future__ import print_function

import sys

sys.path.insert(0, '..')

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.transforms import ToPILImage
from src.blur.data import *
from src.blur.loss import *
from src.eval import *
from src.blur.model import *
from argparse import ArgumentParser
from datetime import datetime
from scipy import misc
from tqdm import tqdm
import os
import numpy as np

from src.blur.train_test import get_test_loader, get_model

def save_result(root_dir, filename, img, gt, out):
    fname = os.path.basename(filename)
    fname = fname[:fname.index('.')]
    infix = ['rst', 'gt', 'src']
    out_path, gt_path, src_path = [os.path.join(root_dir, fname + '_' + f + '.bmp') for f in infix]
    out, gt, src = out.cpu().numpy(), gt.numpy(), np.squeeze(img.numpy())
    # print(out.shape, gt.shape, src.shape)

    out = out * 255
    gt = gt * 255
    misc.imsave(out_path, np.squeeze(out.astype(np.uint8)))
    misc.imsave(gt_path, np.squeeze(gt.astype(np.uint8)))


def get_result_from_output(with_fc, args, outputs):
    out = outputs
    if with_fc:
        out, tag = outputs

    if args.eval_method == 'msrcnn':
        idx = torch.squeeze(tag.data.max(1)[1])
        n, c, h, w = out.size()
        idx = idx.repeat(1, h, w, 1).permute(3, 0, 1, 2)  # 1, h, w, n => n, 1, h, w
        out = out.data.gather(1, idx).sigmoid()
        out[out >= 0.5] = 1
        out[out < 0.5] = 0
        # print(out.shape)
        return out

    if args.eval_method == 'sigmoid':
        return out.data.sigmoid()

    return out.data.max(1)[1]  # 获取下标


def test(model, args):
    # prepare
    if not os.path.exists(args.check_point):
        print("ERROR: models does not exists, {}".format(args.check_point))
        return

    model.load_state_dict(torch.load(args.check_point)['state_dict'])

    if not os.path.exists(args.rst_dir):
        os.makedirs(args.rst_dir)

    if args.cuda:
        model.cuda()

    model.eval()

    with_fc = isinstance(model, FCN2sResnetFC)
    if with_fc:
        ds = SegDatasetFC(args.test_list_file, root_dir=None, transform=SegCompose([]),
                          img_transform=img_transform(),
                          label_transform=label_transform())
    else:
        ds = SegDataset(args.test_list_file, root_dir=None, transform=None, img_transform=img_transform(),
                        label_transform=label_transform())

    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=1)

    filenames = open(args.test_list_file).readlines()

    # iterate datasets
    ytrue_list = []
    ypred_list = []
    for idx, data in tqdm(enumerate(dl), total=len(ds)):  # 因为batch size ==1

        ####################
        if with_fc:
            image, label, tag = data
        else:
            image, label = data
        ###################

        if args.cuda:
            var_image = Variable(image.cuda())

        _start = datetime.now()
        outputs = model(var_image)

        ###################
        if with_fc:
            out, ptag = outputs
            # pdb.set_trace()
            pred = ptag.data.cpu().max(1)[1]
            ypred_list.append(pred)
            ytrue_list.append(tag)
        ###################

        _end = datetime.now()
        # print('time eplased: {}s'.format(_end - _start))

        out = get_result_from_output(with_fc, args, outputs)

        tokens = filenames[idx].split()
        src_path, gt_path = tokens[0], tokens[1]
        save_result(args.rst_dir, src_path.strip(), image, label, out)

    ####################
    if with_fc:
        eval_fc(torch.cat(ypred_list, 0), torch.cat(ytrue_list, 0))
    ####################

    eval_seg_dir(args.rst_dir, args.eval_result_file)


def eval_rst(args):
    model = get_model(args)
    model.cuda()

    test_loader = get_test_loader(args)

    ckpt = torch.load(args.ckpt)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    lplabel, ltlabel, lptag, lttag = [], [], [], []
    ltime = []
    for idx, data in tqdm(enumerate(test_loader)):
        if args.test_data_with_fc:
            image, label, tag = data
        else:
            image, label = data

        var_image = Variable(image.cuda())
        _start = datetime.now()
        plabel = model(var_image)
        _end = datetime.now()
        ltime.append(_end - _start)

        if args.with_fc:
            plabel, ptag = plabel
            pntag = ptag.data.cpu().max(1)[1]
            lptag.append(pntag)
            lttag.append(tag)

        if args.eval_method == 'msrcnn':
            ipred = torch.squeeze(ptag.max(1)[1])
            n, c, h, w = plabel.size()
            ipred = ipred.repeat(1, h, w, 1).permute(3, 0, 1, 2)  # 1, h, w, n => n, 1, h, w
            plabel = plabel.gather(1, ipred).sigmoid().data
            plabel[plabel >= 0.5] = 1
            plabel[plabel < 0.5] = 0

        elif args.eval_method == 'sigmoid':
            plabel = plabel.sigmoid().data
            plabel[plabel >= 0.5] = 1
            plabel[plabel < 0.5] = 0
        else:
            plabel = plabel.data.max(1)[1]

        lplabel.append(plabel.cpu())
        ltlabel.append(label)

    if args.with_fc:
        eval_fc(torch.cat(lptag, 0), torch.cat(lttag, 0))

    return eval_pair_batch(torch.cat(lplabel, 0).numpy(), torch.cat(ltlabel, 0).numpy())


def main(args):
    print(args)

    if args.cmd == 'eval':
        eval_rst(args)

    elif args.cmd == 'test':
        if args.rst_dir and not os.path.exists(args.rst_dir):
            os.mkdir(args.rst_dir)

        model_map = {
            'fcn2s': FCN2s,
            'fcn2s-resnet': FCN2sReset,
            'fcn8s': FCN8s,
            'fcn8s-densenet': FCN8sDenseNet,
            'fcn2s-resnet-fc': FCN2sResnetFC
        }

        if args.model == 'fcn2s-resnet':
            # fixme linshicuoshi
            model = FCN2sReset(type=args.type, n_classes=1, none_classifier=True)
        else:
            model = model_map[args.model](type=args.type)
        test(model, args)


if __name__ == '__main__':
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest='cmd')
    test_parser = subparsers.add_parser('test')
    eval_parser = subparsers.add_parser('eval')

    test_parser.add_argument('--check-point', required=True)
    test_parser.add_argument('--rst-dir', required=True)
    test_parser.add_argument('--test-list-file', required=True)
    test_parser.add_argument('--cuda', type=bool, default=True)
    test_parser.add_argument('--model', required=True)
    test_parser.add_argument('--type')
    test_parser.add_argument('--eval-method', default='default')

    eval_parser.add_argument('--rst-dir', required=False)
    eval_parser.add_argument('--model', required=False)
    eval_parser.add_argument('--train-list-file', required=False)
    eval_parser.add_argument('--test-data-file', required=False)
    eval_parser.add_argument('--test-list-file', required=False)
    eval_parser.add_argument('--test-from-file', type=bool, default=False, required=False)
    eval_parser.add_argument('--lr-policy', default='multi')
    eval_parser.add_argument('--eval-method', default='not msrcnn')
    eval_parser.add_argument('--type', default='resnet152')
    eval_parser.add_argument('--model-root', default='./models')
    eval_parser.add_argument('--vis-port', type=int, default=8097)
    eval_parser.add_argument('--resume', type=bool, default=False)
    eval_parser.add_argument('--ckpt')
    eval_parser.add_argument('--with-fc', type=bool, default=False)
    eval_parser.add_argument('--with-bn', type=bool, default=False)
    eval_parser.add_argument('--pretrained', type=bool, default=False)
    eval_parser.add_argument('--batch-size', type=int, default=16)
    eval_parser.add_argument('--thresh-epoch', type=int, default=30)
    eval_parser.add_argument('--none-pretrained-first', type=bool, default=False)
    eval_parser.add_argument('--test-batch-size', type=int, default=3)
    eval_parser.add_argument('--none-classifier', type=bool, default=False)
    eval_parser.add_argument('--n-epoch', type=int, default=100)
    eval_parser.add_argument('--select-data', type=bool, default=False)
    eval_parser.add_argument('--data-selected', type=str, default=None)
    eval_parser.add_argument('--weighted-sampler', type=bool, default=False)
    eval_parser.add_argument('--test-data-with-fc', type=bool, default=False)
    eval_parser.add_argument('--weight-path', default=None)
    eval_parser.add_argument('--multi-str', default=None)
    eval_parser.add_argument('--power', default=0.5, type=float)
    eval_parser.add_argument('--num-classes', default=2, type=int)
    eval_parser.add_argument('--weight-of-motion', type=float, default=None)
    eval_parser.add_argument('--with-edge', type=bool, default=False)
    eval_parser.add_argument('--only-edge', type=bool, default=False)

    main(parser.parse_args())
