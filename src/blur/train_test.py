# coding: utf-8

from __future__ import print_function
from __future__ import absolute_import

import torch
import os
import time
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, ExponentialLR, LambdaLR
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser
from datetime import datetime
from visdom import Visdom
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from datetime import datetime


from src.blur.util import vis_line
from src.eval import eval_fc, eval_pair_batch
from src.blur.loss import *
from src.blur.model import *
from src.blur.data import *

from src.blur.data import load_img, load_gray_img


from src.pytorch.backbone.seg import DeepLab_v3_plus
from torch.utils.data.sampler import WeightedRandomSampler

"""
训练和测试一起进行
"""

dict_class = {
    'motion': 0,
    'out_of_focus': 1
}


class Layer(object):
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.params = []

    def __str__(self):
        return 'name: {}, type: {}'.format(self.name, self.type)


# todo like schedule sample
def loss_and_backward(model, criterion, optimizer, data, args):
    optimizer.zero_grad()
    if not args.with_fc:
        # images, labels = data
        # images, labels = Variable(images.cuda()), Variable(labels.cuda())
        data = [Variable(d.cuda()) for d in data]
        images = data[0]
        if len(data) == 2:
            labels = data[1]
        else:
            labels = data[1:]
        outputs = model(images)
        loss = criterion(outputs, labels)

    else:
        images, labels, tags = data
        images, labels, tags = Variable(images.cuda()), Variable(labels.cuda()), Variable(tags.cuda())
        pre_labels, pre_tags = model(images)
        loss = criterion(pre_labels, labels, pre_tags, tags)

    loss.backward()
    optimizer.step()
    return loss


def train(model, criterion, optimizer, train_loader, idx_epoch, args):
    model.train()
    # pbar = tqdm(train_loader, total=len(train_loader))
    _epoch_loss = []
    for idx_batch, batch in enumerate(train_loader):
        loss = loss_and_backward(model, criterion, optimizer, batch, args)
        _epoch_loss.append(loss.data[0])
        _mean_loss = sum(_epoch_loss) / len(_epoch_loss)
        print("[{}] epoch {:3} step: {:3} loss: {:.7f} mean loss: {:.7f}".format(
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            idx_epoch, idx_batch, _epoch_loss[-1],
            _mean_loss))
        # pbar.set_description("step: {:4} loss: {:.7f} mean loss: {:.7f}".format(idx_batch, _epoch_loss[-1], _mean_loss))

    return _epoch_loss


def test_with_edge(model, test_loader, args):
    model.eval()
    lplabel, ltlabel, lpedge, ltedge = [], [], [], []
    ltime = []
    for idx, data in enumerate(test_loader):

        if args.only_edge and args.test_data_with_fc:
            image, edge, tag = data

        elif args.test_data_with_fc:
            image, label, tag, edge = data
        else:
            image, label, edge = data

        var_image = Variable(image.cuda())
        _start = datetime.now()

        seg_pred = None
        edge_pred = None
        if not args.only_edge:
            seg_pred, edge_pred = model(var_image)
        else:
            edge_pred = model(var_image)
        _end = datetime.now()
        ltime.append(_end - _start)

        if seg_pred is not None:
            seg_pred = seg_pred.sigmoid().data
            seg_pred[seg_pred >= 0.5] = 1
            seg_pred[seg_pred < 0.5] = 0
            lplabel.append(seg_pred.cpu())
            ltlabel.append(label)

        if edge_pred is not None:
            edge_pred = edge_pred.sigmoid().data
            edge_pred[edge_pred >= 0.5] = 1
            edge_pred[edge_pred < 0.5] = 0
            lpedge.append(edge_pred.cpu())
            ltedge.append(edge)

    eval_fc(torch.cat(lpedge, 0).view((-1,)).float(), torch.cat(ltedge, 0).view((-1,)).float())

    if not args.only_edge:
        return eval_pair_batch(torch.cat(lplabel, 0).numpy(), torch.cat(ltlabel, 0).numpy())

    return None


def test(model, test_loader, args):
    model.eval()
    lplabel, ltlabel, lptag, lttag = [], [], [], []
    ltime = []
    for idx, data in enumerate(test_loader):
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


def get_model(args):
    model_dict = {
        'fcn2s': FCN2s,
        'fcn2s-resnet': FCN2sReset,
        'fcn8s': FCN8s,
        'fcn8s-densenet': FCN8sDenseNet,
        'fcn2s-resnet-fc': FCN2sResnetFC,
        'segnet': SegNet,
        'unet': UNet,
        'deepsp': DeepSupervisedFCN,
        'pspnet': PSPNet,
        'fcn2s-resnet-deconv': FCN2sDeconv,
        'fcn2s-hed': DeepHEDFCN2s,
        'hed-resnet': EdgeHEDResNet,
        'hed-vgg': EdgeHED,
        'fcn2s-hed-resnet-back': DeepHEDFCN2sBackResNet
    }

    if args.model=='fcn2s-hed-resnet-back':
        return DeepHEDFCN2sBackResNet(pretrained=args.pretrained, resnet_type=args.type)
    elif args.model == 'hed-vgg':
        return EdgeHED(pretrained=args.pretrained)
    elif args.model == 'hed-resnet':
        return EdgeHEDResNet(pretrained=args.pretrained, resnet_type=args.type)
    elif args.model == 'fcn2s-hed':
        return DeepHEDFCN2s(pretrained=args.pretrained, with_bn=args.with_bn, n_classes=1)
    elif args.model == 'segnet':
        return SegNet(pretrained=args.pretrained, with_bn=args.with_bn)
    elif args.model == 'fcn2s':
        # init_weight_with_caffe = {'method': 'seq', 'weight_path':
        #  '/home/adam/Gits/blur-detection/src/fcn32s-fc-pascal-all.pkl'}
        init_weight_with_caffe = {'method': 'classifier',
                                  'weight_path': '/home/adam/Gits/blur-detection/src/fcn32s-fc-pascal.pth'}
        return FCN2s(pretrained=args.pretrained, init_weight_with_caffe=init_weight_with_caffe)
    elif args.model == 'unet' or args.model == 'unetv2':
        return UNet(pretrained=args.pretrained, with_bn=args.with_bn)
    elif args.model == 'fcn2s-resnet':
        return FCN2sReset(type=args.type, none_classifier=args.none_classifier)
    elif args.model == 'fcn2s-resnet-deconv':
        return FCN2sDeconv(type=args.type, none_classifier=args.none_classifier, pretrained=args.pretrained,
                           n_classes=args.num_classes)
        # 以下代码是用通过sigmoid生成灰度结果
        # return FCN2sReset(type=args.type, n_classes=1, none_classifier=args.none_classifier)
    elif args.model == 'fcn2s-resnet-srcnn':
        return SRCNNResNetFCN2s(weight_path=args.weight_path)
    elif args.model == 'deepsp':
        return DeepSupervisedFCN(num_classes=args.num_classes, with_bn=args.with_bn, pretrained=args.pretrained)
    elif args.model == 'pspnet':
        return PSPNet(3, model_type=args.type)
    elif args.model == 'deeplab-v3p':
        return DeepLab_v3_plus.DeepLabv3_plus(2, pretrained=args.pretrained)

    return model_dict[args.model](type=args.type)


def get_train_loader(args):
    if args.with_fc:
        dataset = SegDatasetFC.instance(args.train_list_file)
    elif args.select_data:
        dataset = SeperatableSegDataset.instance(args.train_list_file, only_select=args.data_selected)
    else:
        dataset = SegDataset.instance(args.train_list_file,
                                      img_transform=nuceli_img_transform(),
                                      weight_of_motion=args.weight_of_motion,
                                      with_edge=args.with_edge,
                                      only_edge=args.only_edge)

    if args.weighted_sampler:
        sampler = WeightedRandomSampler(dataset.get_weight(), len(dataset))
        return DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, shuffle=False, num_workers=4)

    else:
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)


def get_test_loader(args, **kwargs):
    if args.test_from_file:
        dataset = PILDictDataset.instance(args.test_data_file, test=True, with_fc=args.test_data_with_fc,
                                          only_select=(None if not args.select_data else args.data_selected),
                                          with_edge=args.with_edge,
                                          only_edge=args.only_edge)
    else:
        if args.with_fc:
            dataset = SegDatasetFC.instance(args.test_list_file)
        elif args.select_data:
            dataset = SeperatableSegDataset.instance(args.test_list_file, only_select=args.data_selected)
        else:
            dataset = SegDataset.instance(args.test_list_file, img_transform=img_transform(), **kwargs)

    return DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=0)


def get_criterion(args):
    loss_dict = {
        'default': CrossEntropyLoss,
        'ce_fc': CrossEntropyLossFC,
        'weighted_c': WeightedMyLoss,
        'ms_fc': MSRNNLossFC,
        'bce': BCELoss,
        'deepsp': DeepSupervisedLoss,
        'pspnet': PSPNetLoss,
        'fcn2s-hed': DeepHEDFCN2sLoss,
        'hed': HedLoss,
    }

    if args.loss == 'deepsp-focal':
        return DeepSupervisedLoss(loss_fn=FocalLoss2d(gamma=1, logit=False))

    if args.loss == 'deepsp-iou':
        return DeepSupervisedLoss(loss_fn=IOULoss2d(num_classes=args.num_classes))

    if args.loss == 'deepsp-al':
        return DeepSupervisedLoss(loss_fn=AL2Loss2d(num_classes=args.num_classes))

    return loss_dict[args.loss]()


def get_optimizer(model, args):
    parameters = model.parameters()
    if args.none_pretrained_first:
        if hasattr(model, 'none_pretrained_parameters'):
            print('add none pretrained_parameters')
            parameters = model.none_pretrained_parameters()

    if args.optimizer == 'sgd':
        optimizer = SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                        nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)

    else:
        optimizer = SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                        nesterov=True)

    return optimizer


def update_optimizer(model, optimizer, idx_epoch, args):
    if idx_epoch == args.thresh_epoch and hasattr(model, 'pretrained_parameters'):
        print("-------------add pretrained parameter to train--------------")
        optimizer.add_param_group({'params': model.pretrained_parameters()})


def get_lr_scheduler(optimizer, args):
    if args.lr_policy == 'multi':
        if args.multi_str is None:
            return MultiStepLR(optimizer, [30, 60, 90, ])
        else:
            steps = list(int(m) for m in args.multi_str.split(', '))
            print(steps)
            return MultiStepLR(optimizer, steps)

    elif args.lr_policy == 'auto':
        return ReduceLROnPlateau(optimizer, min_lr=1e-7)
    elif args.lr_policy == 'poly':
        return LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / float(args.epoch) ** args.power))
    else:
        return None

def main(args):
    if not os.path.exists(os.path.join(args.model_root, args.exp_name)):
        os.makedirs(os.path.join(args.model_root, args.exp_name))

    model = get_model(args)
    model.cuda()
    # print(model)

    if args.visdom:
        vis = Visdom(port=args.vis_port)

    criterion = get_criterion(args)
    optimizer = get_optimizer(model, args)
    lr_policy = get_lr_scheduler(optimizer, args)

    start_epoch = 0
    if args.resume:
        if os.path.exists(args.ckpt):
            ckpt = torch.load(args.ckpt)
            model.load_state_dict(ckpt['state_dict'])
            update_optimizer(model, optimizer, ckpt['epoch'], args)
            optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt['epoch'] + 1
        else:
            print("no check point find", args.ckpt)

    train_loader, test_loader = get_train_loader(args), get_test_loader(args)

    total_train_loss = []
    for idx_epoch in range(start_epoch, args.n_epoch + 1):
        update_optimizer(model, optimizer, idx_epoch, args)
        _train_loss = train(model, criterion, optimizer, train_loader, idx_epoch, args)
        _train_loss = np.asarray(_train_loss)
        if args.visdom:
            vis_line(_train_loss, vis, 'epoch {:03}'.format(idx_epoch), subject='loss_' + args.exp_name)

        if not (args.with_edge or args.only_edge):
            _, _, miu, _ = test(model, test_loader, args)
            total_train_loss.append(_train_loss)

            if lr_policy is not None:
                if lr_policy == 'sgd':
                    lr_policy.step(epoch=idx_epoch + 1)

                elif lr_policy == 'auto':
                    lr_policy.step(miu, epoch=idx_epoch + 1)

        else:
            test_with_edge(model, test_loader, args)

        if not args.save_best:
            if args.epoch_save > 0 and idx_epoch > 0 and idx_epoch % args.epoch_save == 0:
                fname = 'epoch-{:03}.pth'.format(idx_epoch)
                fpath = os.path.join(args.model_root, args.exp_name, fname)
                if args.save:
                    torch.save({
                        'epoch': idx_epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }, fpath)

                print('save: {} at epoch: {}'.format(fname, idx_epoch))

        else:
            pass

    if args.visdom:
        vis_line(np.concatenate(tuple(total_train_loss), axis=0), vis, 'total', subject='loss_' + args.exp_name)


def gen_data(test_file, rst_file):
    """
    生成测试集 提高效率
    :param test_file:
    :param rst_file:
    :return:
    """

    def read_accord_list(filenames, gray=False):
        images = []
        for filename in tqdm(filenames):
            img = load_img(filename) if not gray else load_gray_img(filename)
            images.append(img)

        return images

    df = pd.read_csv(test_file, index_col=None, header=None, delim_whitespace=True)
    print(df.shape)

    if len(df.columns) == 3:
        tags = [dict_class[f] for f in df.iloc[:, 2]]
        images, gts = read_accord_list(df.iloc[:, 0]), read_accord_list(df.iloc[:, 1], True)
        to_save = {'images': images, 'gts': gts, 'tags': tags}
        torch.save(to_save, rst_file)

    elif len(df.columns) == 2:
        images, gts = read_accord_list(df.iloc[:, 0]), read_accord_list(df.iloc[:, 1], True)
        to_save = {'images': images, 'gts': gts}
        torch.save(to_save, rst_file)


# gen_data('/home/adam/Gits/blur-seg/grid_db/val_pair_fc.txt', 'test_fc.pth') # 300M
# gen_data('/home/adam/Gits/blur-seg/grid_db/train_pair_fc.txt', 'train_fc.pth')  # 生成的train 文件有2.6g

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--exp-name', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--train-list-file', required=True)
    parser.add_argument('--test-data-file', required=False)
    parser.add_argument('--test-list-file', required=False)
    parser.add_argument('--test-from-file', type=bool, default=False, required=False)

    parser.add_argument('--loss', required=True)
    parser.add_argument('--lr-policy', default='multi')
    parser.add_argument('--optimizer', default='sgd')
    parser.add_argument('--eval-method', default='not msrcnn')
    parser.add_argument('--type', default='resnet152')

    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--weight-decay', default=1e-6, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)

    parser.add_argument('--model-root', default='./models')
    parser.add_argument('--vis-port', type=int, default=8097)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--ckpt')
    parser.add_argument('--with-fc', type=bool, default=False)
    parser.add_argument('--with-bn', type=bool, default=False)
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--thresh-epoch', type=int, default=30)
    parser.add_argument('--none-pretrained-first', type=bool, default=False)
    parser.add_argument('--test-batch-size', type=int, default=3)
    parser.add_argument('--none-classifier', type=bool, default=False)

    parser.add_argument('--n-epoch', type=int, default=100)
    parser.add_argument('--save-best', type=bool, default=False)
    parser.add_argument('--epoch-save', type=int, default=20)

    ## 选择样本 SeperableDataset
    parser.add_argument('--select-data', type=bool, default=False)
    parser.add_argument('--data-selected', type=str, default=None)
    parser.add_argument('--weighted-sampler', type=bool, default=False)
    parser.add_argument('--test-data-with-fc', type=bool, default=False)

    ## fcn2s srcnn
    parser.add_argument('--weight-path', default=None)

    parser.add_argument('--visdom', default=False, type=bool)
    parser.add_argument('--multi-str', default=None)
    parser.add_argument('--power', default=0.5, type=float)

    parser.add_argument('--num-classes', default=2, type=int)

    parser.add_argument('--weight-of-motion', type=float, default=None)
    parser.add_argument('--with-edge', type=bool, default=False)
    parser.add_argument('--only-edge', type=bool, default=False)

    args = parser.parse_args()
    print(args)
    main(args)
