# coding: utf-8
from __future__ import print_function

from argparse import ArgumentParser
from datetime import datetime

from src.blur.model import *
from src.blur.data import SegDataset, SegDatasetFC
from src.blur.loss import *
from src.blur.util import *
from visdom import Visdom

from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable
from torch.optim import SGD
from torch.optim import lr_scheduler
import torch

from sklearn.model_selection import KFold
import numpy as np
import os
from tensorboardX import SummaryWriter

PRETRAINED_FCN8S_PROTOFILE = ''
PRETRAINED_FCN8S_CAFFEMODEL = ''

def get_criterion(args):
    if args.loss == 'default':
        return CrossEntropyLoss()

    elif args.loss == 'fc':
        return CrossEntropyLossFC()

    elif args.loss == 'weighted':
        return WeightedMyLoss()

    elif args.loss == 'bceloss':
        return MSRNNLossFC()


def cal_loss_backward(iter_data, args, model, criterion, opt, train=True):
    if train==False:
        torch.no_grad()

    loss = None
    opt.zero_grad()
    if len(iter_data) == 2:
        images, labels = iter_data
        if args.cuda:
            images, labels = images.cuda(), labels.cuda()

        images, labels = Variable(images), Variable(labels)
        outputs = model(images)
        loss = criterion(outputs, labels)

    elif len(iter_data) == 3:
        images, labels, tags = iter_data
        if args.cuda:
            images, labels, tags = images.cuda(), labels.cuda(), tags.cuda()

        images, labels, tags = Variable(images), Variable(labels), Variable(tags)
        pre_labels, pre_tags = model(images)
        loss = criterion(pre_labels, labels, pre_tags, tags)

    if train:
        loss.backward()
        opt.step()

    return loss


def get_dataset(model, args):
    if isinstance(model, FCN2sResnetFC):
        ds = SegDatasetFC.instance(args.train_list_file)
    else:
        ds = SegDataset.instance(args.train_list_file)

    return ds


def get_lr_scheduler(opt):
    return lr_scheduler.MultiStepLR(opt, milestones=[30, 40])


def train(model, args, exp_name=None):
    if args.cuda:
        model.cuda()

    model.train()

    ds = get_dataset(model, args)
    K = 10
    n_val = int(len(ds) // K)
    n_train = len(ds) - n_val
    trainset, valset = random_split(ds, [n_train, n_val])

    dl = DataLoader(trainset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.n_workers)
    vl = DataLoader(valset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.n_workers)

    vis = Visdom(port=args.vis_port)

    criterion = get_criterion(args)
    opt = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)

    start_epoch = 0

    if args.resume:
        if os.path.exists(args.check_point):
            cp = torch.load(args.check_point)
            model.load_state_dict(cp['state_dict'])
            opt.load_state_dict(cp['optimizer'])
            start_epoch = cp['epoch'] + 1

        else:
            print("no check point find: ", args.check_point)

    print(model)

    _start_time = datetime.now()
    _train_loss = []

    lr_policy = get_lr_scheduler(opt)

    if os.path.exists(os.path.join(args.model_root, args.exp)):
        os.makedirs(os.path.join(args.model_root, args.exp))

    writer_x = 0
    epoch_x = 0

    for i_epoch in range(start_epoch, args.epoch + 1):
        _epoch_loss = []
        _val_loss = []
        lr_policy.step(epoch=i_epoch + 1)
        # train
        for i_batch, iter_data in enumerate(dl):
            loss = cal_loss_backward(iter_data, args, model, criterion, opt)
            _epoch_loss.append(loss.data[0])
            if args.step_print > 0 and i_batch % args.step_print == 0:
                print("epoch: {:03} step: {:4} loss: {:.7f} mean loss: {:.7f}".format(i_epoch, i_batch, loss.data[0],
                                                                                      sum(_epoch_loss) / len(
                                                                                          _epoch_loss)))

        #  validation
        for i_batch, iter_data in enumerate(vl):
            loss = cal_loss_backward(iter_data, args, model, criterion, opt, train=False)
            _val_loss.append(loss.data[0])

            if args.step_print > 0 and i_batch % args.step_print == 0:
                print("validation: epoch: {:03} step: {:4} loss: {:.7f} mean loss: {:.7f}".format(i_epoch, i_batch, loss.data[0],
                                                                                      sum(_val_loss) / len(
                                                                                          _val_loss)))


        if args.epoch_save > 0 and i_epoch % args.epoch_save == 0:
            fname = 'epoch-{:03}.pth'.format(i_epoch)
            fpath = os.path.join(args.model_root, args.exp, fname)
            if args.save:
                torch.save({
                    'epoch': i_epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': opt.state_dict()
                }, fpath)
            print('save: {} at epoch: {}'.format(fname, i_epoch))

        print('epoch: {:03}>> mean loss: {:.7f}'.format(i_epoch, sum(_epoch_loss) / len(_epoch_loss)))
        print('epoch: {:03}>> validation mean loss: {:.7f}'.format(i_epoch, sum(_val_loss) / len(_val_loss)))

        _train_loss += _epoch_loss
        vis_line(np.asarray(_epoch_loss), vis, 'epoch {:03}'.format(i_epoch), subject='loss_' + exp_name)

    _end_time = datetime.now()
    print("train finished start: {} end {} duration: {}s".format(_start_time, _end_time, _end_time - _start_time))
    vis_line(np.asarray(_train_loss), vis, 'train loss', subject='loss_' + exp_name)

def train_kflod(model, args, exp_name=None):

    writer = SummaryWriter(comment="_epoch100_mean")
    writer1 = SummaryWriter(comment="_epoch100_k1")
    writer2 = SummaryWriter(comment="_epoch100_k2")
    writer3 = SummaryWriter(comment="_epoch100_k3")
    writerlist = [writer1, writer2, writer3]

    start_epoch = 0

    _start_time = datetime.now()
    _train_loss = []

    if os.path.exists(os.path.join(args.model_root, args.exp)):
        os.makedirs(os.path.join(args.model_root, args.exp))

    mean_train_loss_k_total = []
    mean_val_loss_k_total = []

    ds = get_dataset(model, args)
    K = 3
    kf = KFold(n_splits=K, shuffle=True)
    model = None

    for i, (train_index, test_index) in enumerate(kf.split(ds)):

        model = FCN2sResnetFC(type=args.type)
        if args.cuda:
            model.cuda()

        model.train()
        criterion = get_criterion(args)
        opt = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
        lr_policy = get_lr_scheduler(opt)

        epoch_x = 0

        mean_train_loss_k = []
        mean_val_loss_k = []

        train = torch.utils.data.Subset(ds, train_index)
        test = torch.utils.data.Subset(ds, test_index)

        dl = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers,
                                                  pin_memory=True)
        vl = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers,
                                                 pin_memory=True)

        print('Fold : {}, train : {}, test : {}'.format(i + 1, len(dl.dataset), len(vl.dataset)))

        writer_x = 0
        for i_epoch in range(start_epoch, args.epoch + 1):

            lr_policy.step(epoch=i_epoch + 1)
            epoch_x += 1

            _epoch_loss = []
            _val_loss = []
            currentWriter = writerlist[i]
            # train
            for i_batch, iter_data in enumerate(dl):
                loss = cal_loss_backward(iter_data, args, model, criterion, opt)
                _epoch_loss.append(loss.data[0])
                if args.step_print > 0 and i_batch % args.step_print == 0:
                    print("Fold{} epoch: {:03} step: {:4} loss: {:.7f} mean loss: {:.7f}".format(i+1, i_epoch, i_batch, loss.data[0],
                                                                                          sum(_epoch_loss) / len(
                                                                                              _epoch_loss)))
                    writer_x += args.step_print
                    mean_loss = sum(_epoch_loss) / len(_epoch_loss)
                    currentWriter.add_scalar("Fold_{} Mean Loss of Train".format(i+1), mean_loss, writer_x)

            #  validation
            for i_batch, iter_data in enumerate(vl):
                loss = cal_loss_backward(iter_data, args, model, criterion, opt, train=False)
                _val_loss.append(loss.data[0])
                if args.step_print > 0 and i_batch % args.step_print == 0:
                    print("Fold{} validation: epoch: {:03} step: {:4} loss: {:.7f} mean loss: {:.7f}".format(i+1, i_epoch, i_batch, loss.data[0],
                                                                                          sum(_val_loss) / len(
                                                                                              _val_loss)))

            print('Fold{} epoch: {:03}>> mean loss: {:.7f}'.format(i+1, i_epoch, sum(_epoch_loss) / len(_epoch_loss)))
            print('Fold{} epoch: {:03}>> validation mean loss: {:.7f}'.format(i+1, i_epoch, sum(_val_loss) / len(_val_loss)))

            mean_loss = sum(_epoch_loss) / len(_epoch_loss)
            mean_val_loss = sum(_val_loss) / len(_val_loss)
            currentWriter.add_scalar("Mean Loss of Train KFold", mean_loss, epoch_x)
            currentWriter.add_scalar("Mean Loss of Validation KFold", mean_val_loss, epoch_x)

            mean_train_loss_k.append(mean_loss)
            mean_val_loss_k.append(mean_val_loss)
            _train_loss += _epoch_loss

        mean_train_loss_k_total += mean_train_loss_k
        mean_val_loss_k_total += mean_val_loss_k

    writer.close()
    writer1.close()
    writer2.close()
    writer3.close()


def main(args):
    print(args)

    if args.model_root and not os.path.exists(args.model_root):
        os.mkdir(args.model_root)

    model_map = {
        'fcn2s': FCN2s,
        'fcn2s-resnet': FCN2sReset,
        'fcn8s': FCN8s,
        'fcn8s-densenet': FCN8sDenseNet,
        'fcn2s-resnet-fc': FCN2sResnetFC
    }

    model = model_map[args.model]

    exp_name = args.exp if args.exp is not None else '{}_{}'.format(args.model, args.type)

    train(model(type=args.type), args, exp_name)
    # train_kflod(model(type=args.type), args, exp_name)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--vis-port', type=int, default=8097)
    parser.add_argument('--train-list-file', required=True)
    parser.add_argument('--model', default='fcn2s')
    parser.add_argument('--type')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=5)
    parser.add_argument('--n-workers', type=int, default=4)
    parser.add_argument('--epoch-save', type=int, default=10)
    parser.add_argument('--step-print', type=int, default=20)
    parser.add_argument('--crop-size', type=int, default=224)
    parser.add_argument('--model-root', default='./models')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--train-start', type=int, default=0)
    parser.add_argument('--check-point', type=str)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--loss', default='default')
    parser.add_argument('--exp')
    main(parser.parse_args())
