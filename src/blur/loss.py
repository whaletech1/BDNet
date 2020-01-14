# coding: utf-8
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import cv2
import numpy as np

_nuclei_weight = torch.from_numpy(np.asarray([0.58, 3.73])).float().cuda()


def weighted_binary_cross_entropy_with_logits(logits, targets, pos_weight, weight=None, size_average=True, reduce=True):
    if not (targets.size() == logits.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(targets.size(), logits.size()))

    max_val = (-logits).clamp(min=0)
    log_weight = 1 + (pos_weight - 1) * targets
    loss = (1 - targets) * logits + log_weight * ((-logits.abs()).exp().log1p() + max_val)

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()


class HedLoss(nn.Module):
    def __init__(self, weights=None):
        super(HedLoss, self).__init__()
        self.pos_weight = 0.96
        self.weights = weights

    def forward(self, inputs, targets):
        """
        :param inputs: [x6]
        :param targets: None*1*h*w
        :return:
        """
        total = None
        weights = self.weights
        if weights is None:
            weights = [1, ] * len(inputs)

        losses = []
        for pred, weight in zip(inputs, weights):
            _loss = weighted_binary_cross_entropy_with_logits(torch.squeeze(pred.float()), targets.float(), pos_weight=self.pos_weight)
            _p_loss = _loss * weight
            losses.append(_p_loss.data[0])
            if total is None:
                total = _p_loss
            else:
                total += _p_loss

        print("edge loss: " + '\t'.join(['{:.7f}'.format(l) for l in losses]), end='\t')
        return total


class DeepHEDFCN2sLoss(nn.Module):
    def __init__(self, seg_weight=1, edge_weight=1):
        super(DeepHEDFCN2sLoss, self).__init__()
        self.seg_weight = seg_weight
        self.edge_weight = edge_weight

    def forward(self, inputs, targets):
        """
        :param inputs: seg, edge[x6]
        :param targets: seg, edge
        :return:
        """
        seg_pred, edge_pred = inputs
        seg_targets, edge_targets = targets

        seg_loss = F.binary_cross_entropy_with_logits(torch.squeeze(seg_pred), torch.squeeze(seg_targets.float()))
        edge_loss = HedLoss().forward(edge_pred, torch.squeeze(edge_targets.float()))
        print('seg loss: {:.7f}\tedge loss: {:.7f}'.format(seg_loss.data[0], edge_loss.data[0]), end='\t')
        return self.seg_weight * seg_loss + self.edge_weight * edge_loss


# https://github.com/kyle0x54/DiffLossTest/blob/master/loss.py
class AL2Loss2d(nn.Module):
    def __init__(self, num_classes):
        super(AL2Loss2d, self).__init__()
        self.num_classes = num_classes
        self.cosine_loss = nn.CosineEmbeddingLoss()

    def forward(self, inputs, targets):
        embedding_loss = 0.0
        inputs = inputs.transpose(0, 1)
        center_list = []
        for i in range(self.num_classes):
            mask = self.get_mask(targets, i)
            sum_pixel = max(mask.sum(), 1)
            # print sum_pixel
            mask_ = Variable(torch.cuda.FloatTensor(inputs.size()))
            for j in range(inputs.size()[0]):
                mask_[j] = mask
            center = inputs * mask_
            center = torch.sum(center.view(center.size()[0], -1), 1)
            center = center / sum_pixel
            center_list.append(center)

        center_array = Variable(torch.zeros(self.num_classes, inputs.size()[0]), requires_grad=True).cuda()
        item_count = 0
        for center in center_list:
            center_array[item_count] = center
            item_count = item_count + 1
        for i in range(self.num_classes):
            label = Variable(torch.zeros(self.num_classes, ).type(torch.IntTensor)).cuda()
            center_dual = Variable(torch.zeros(self.num_classes, inputs.size()[0]), requires_grad=True).cuda()
            for k in range(self.num_classes):
                center_dual[k] = center_list[i]

            for j in range(self.num_classes):
                if j == i:
                    label[j] = 1
                else:
                    label[j] = -1
            # print label.size()
            # print center_array.size()
            # print center_dual.size()
            embedding_loss += self.cosine_loss(center_array, center_dual, label)
        # print embedding_loss.requires_grad
        return embedding_loss / (self.num_classes * self.num_classes)

    def get_mask(self, targets, i):
        targets_cp = torch.cuda.FloatTensor(targets.size())
        targets_cp.copy_(targets.data)
        if i == 0:
            targets_cp[targets_cp != 0] = 2
            targets_cp[targets_cp == 0] = 1
            targets_cp[targets_cp == 2] = 0
        else:
            targets_cp[targets_cp != i] = 0
            targets_cp[targets_cp == i] = 1

        return targets_cp


class IOULoss2d(nn.Module):
    def __init__(self, num_classes):
        super(IOULoss2d, self).__init__()
        self.num_classes = num_classes
        self.mse_loss = nn.MSELoss()

    def forward(self, inputs, targets):
        iou_loss = 0  # Variable(torch.zeros(1).type(torch.FloatTensor), requires_grad=True).cuda()
        predicts = Variable(inputs.data.max(1)[1])
        inputs = F.softmax(inputs)

        for i in range(self.num_classes):
            union_mask, intersect_mask = self.get_class_loss(predicts, targets, i)
            union = inputs[:, i:i + 1] * Variable(union_mask)
            intersect = inputs[:, i:i + 1] * Variable(intersect_mask)
            union = torch.sum(union.view(union.size()[0], -1), 1)
            intersect = torch.sum(intersect.view(intersect.size()[0], -1), 1)
            label = Variable(torch.ones(union.size()[0], ).type(torch.FloatTensor)).cuda()
            loss_map = intersect / torch.max(union, label)
            class_iou_loss = self.mse_loss(loss_map, label)
            iou_loss += class_iou_loss
            # print "Class %d: %.4f" % (i, class_iou_loss.data[0])

        return iou_loss

    def get_class_loss(self, predicts, targets, i):
        predicts_cp = torch.cuda.FloatTensor(predicts.size())
        targets_cp = torch.cuda.FloatTensor(targets.size())
        predicts_cp.copy_(predicts.data)
        targets_cp.copy_(targets.data)
        if i == 0:
            predicts_cp[predicts_cp != 0] = 2
            predicts_cp[predicts_cp == 0] = 1
            predicts_cp[predicts_cp == 2] = 0
            targets_cp[targets_cp != 0] = 2
            targets_cp[targets_cp == 0] = 1
            targets_cp[targets_cp == 2] = 0
        else:
            predicts_cp[predicts_cp != i] = 0
            predicts_cp[predicts_cp == i] = 1
            targets_cp[targets_cp != i] = 0
            targets_cp[targets_cp == i] = 1

        union = predicts_cp + targets_cp
        intersect = predicts_cp * targets_cp
        union[union == 2] = 1
        return union, intersect


class FocalLoss(nn.Module):
    def __init__(self, y):
        super(FocalLoss, self).__init__()
        self.y = y

    def forward(self, output, target):
        P = F.softmax(output)
        f_out = F.log_softmax(output)
        Pt = P.gather(1, torch.unsqueeze(target, 1))
        focus_p = torch.pow(1 - Pt, self.y)
        alpha = 0.25
        nll_feature = -f_out.gather(1, torch.unsqueeze(target, 1))
        weight_nll = alpha * focus_p * nll_feature
        loss = weight_nll.mean()
        return loss


class FocalLoss2d(nn.Module):
    """
    https://github.com/doiken23/focal_segmentation
    """

    def __init__(self, gamma=0, weight=None, size_average=True, logit=False):
        super(FocalLoss2d, self).__init__()

        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average
        self.logit = logit

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.contiguous().view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2)).squeeze()
        if target.dim() == 4:
            target = target.contiguous().view(target.size(0), target.size(1), -1)
            target = target.transpose(1, 2)
            target = target.contiguous().view(-1, target.size(2)).squeeze()
        elif target.dim() == 3:
            target = target.view(-1)
        else:
            target = target.view(-1, 1)

        # compute the negative likelyhood

        if self.logit:
            logpt = -F.binary_cross_entropy(F.sigmoid(input), target)
        else:
            logpt = -F.cross_entropy(input, target)

        pt = torch.exp(logpt)

        # compute the loss
        loss = -((1 - pt) ** self.gamma) * logpt

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class WeightedBCEWithLogitsLoss(torch.nn.Module):
    def __init__(self, pos_weight, weight=None, size_average=True, reduce=True):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, input, target):
        pos_weight = Variable(self.pos_weight) if not isinstance(self.pos_weight, Variable) else self.pos_weight
        if self.weight is not None:
            weight = Variable(self.weight) if not isinstance(self.weight, Variable) else self.weight
            return weighted_binary_cross_entropy_with_logits(input, target,
                                                             pos_weight,
                                                             weight=weight,
                                                             size_average=self.size_average,
                                                             reduce=self.reduce)
        else:
            return weighted_binary_cross_entropy_with_logits(input, target,
                                                             pos_weight,
                                                             weight=None,
                                                             size_average=self.size_average,
                                                             reduce=self.reduce)


class DeepSupervisedLoss(nn.Module):
    """deepsp"""

    def __init__(self, loss_fn=None):
        super(DeepSupervisedLoss, self).__init__()
        self.loss_fn = loss_fn

    def _loss(self, output, pred):
        if weight is not None:
            assert len(_nuclei_weight) == 2
            loss = _nuclei_weight[1] * (pred * torch.log(output)) + _nuclei_weight[0] * (
                    (1 - pred) * torch.log(1 - output))
        else:
            loss = pred * torch.log(output) + (1 - pred) * torch.log(1 - output)

        return torch.neg(torch.mean(loss))

    def forward(self, outputs, targets):
        if self.loss_fn is None:
            losses = list([F.binary_cross_entropy(F.sigmoid(pred), targets.float()) for pred in outputs])
            # losses = list([self._loss(F.sigmoid(pred), targets.float()) for pred in outputs])
            # losses = list(
            #     [weighted_binary_cross_entropy_with_logits(torch.squeeze(F.sigmoid(pred)), torch.squeeze(targets.float()),
            #                                                pos_weight=7) for
            #      pred in outputs])
        else:
            losses = list([self.loss_fn(pred, torch.squeeze(targets)) for pred in outputs])
        total_loss = sum(losses)
        return total_loss


class PSPNetLoss(nn.Module):
    def __init__(self, alpha=0.4):
        super(PSPNetLoss, self).__init__()
        self.alpha = alpha

    def forward(self, outputs, targets):
        final_outs, aux_outs = outputs
        loss_final = F.cross_entropy(final_outs, targets)
        loss_aux = F.cross_entropy(aux_outs, targets)
        return loss_final + self.alpha * loss_aux


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        # weight.requires_grad = False
        self.loss = nn.CrossEntropyLoss(weight=_nuclei_weight)

    def forward(self, out, targets):
        return self.loss(out, targets)


class ColorSmoothCrossEntroyLoss(nn.Module):
    """
    SeNet: Structured Edge Network for Sea–Land Segmentation
    """

    def __init__(self):
        super(ColorSmoothCrossEntroyLoss, self).__init__()

    def forward(self, *input):
        pass


class SmoothCrossEntroyLoss(nn.Module):
    """
    Topology Aware Fully Convolutional Networks For Histology Gland Segmentation

    4 connected neighbour
    """

    def __init__(self):
        super(SmoothCrossEntroyLoss, self).__init__()

    def forward(self, *input):
        pass


class BCELoss(nn.Module):
    """使用的時候不要在網絡中進行sigmoid"""

    def __init__(self):
        super(BCELoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, outs, targets):
        outs, targets = outs.squeeze().float(), targets.squeeze().float()
        return self.loss(outs, targets)


class LocalRegularizedCEL(nn.Module):
    """
    SeNet: Structured Edge Network for Sea–Land Segmentation
    """

    def __init__(self):
        super(LocalRegularizedCEL, self).__init__()

    def forward(self, inputs, logits):
        pass

    @staticmethod
    def laplacian(self, x):
        """
        laplacian for batch x
        :param self:
        :param x: batch of images
        :return:
        """
        pass


class_weight = torch.from_numpy(np.asarray([1.5, 0.75])).float().cuda()


class CrossEntropyLossFC(nn.Module):
    def __init__(self):
        super(CrossEntropyLossFC, self).__init__()
        self.loss_seg = nn.CrossEntropyLoss(weight=None)
        self.loss_class = nn.CrossEntropyLoss(weight=class_weight)

    def forward(self, outputs, targets, tags, ptags):
        return 0.8 * self.loss_seg(outputs, targets) + 0.2 * self.loss_class(tags, ptags)


class MSRNNLossFC(nn.Module):
    def __init__(self):
        super(MSRNNLossFC, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.cross_class = nn.CrossEntropyLoss(weight=class_weight)

    def forward(self, outputs, targets, ptags, tags):
        idx = torch.squeeze(ptags.max(1)[1])
        n, c, h, w = outputs.size()
        idx = idx.repeat(1, h, w, 1).permute(3, 0, 1, 2)  # 1, h, w, n => n, 1, h, w
        seg_pred = outputs.gather(1, idx)
        seg_target = targets.float().unsqueeze(1)

        return 0.8 * self.bce(seg_pred, seg_target) + 0.2 * self.cross_class(ptags, tags)


class WeightedMyLoss(nn.Module):
    """
    曾强边界
    """

    def __init__(self):
        super(WeightedMyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(weight=None, size_average=False, reduce=False)

    def forward(self, outputs, targets):
        losses = self.loss(outputs, targets)

        weights = []
        for target in targets:
            weights.append(self.weight(target))

        weights = Variable(torch.FloatTensor(np.asarray(weights))).cuda()

        weighted = losses.mul(weights)
        return weighted.sum() / weights.sum()

    def weight(self, target):
        target = target.cpu().data.numpy().astype(np.uint8, copy=False)
        kernel_size = 7
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2 * kernel_size + 1, 2 * kernel_size + 1))
        dilate = cv2.dilate(target, kernel=kernel)
        erode = cv2.erode(target, kernel=kernel)
        weight = cv2.bitwise_xor(dilate, erode)
        weight = weight.astype(np.float32) + 1.0
        weight = weight / np.sum(weight) * np.size(weight)  # reweight
        return weight


if __name__ == '__main__':
    predict = torch.FloatTensor(2, 2, 256, 256).random_(2)
    target = torch.LongTensor(2, 256, 256).random_(2)
    print(predict.size(), target.size())

    # loss = CrossEntropyLoss()
    # ls = loss(Variable(predict).cuda(), Variable(target).cuda())
    # print(ls)

    loss = FocalLoss2d(gamma=0, logit=False)
    ls = loss(Variable(predict).cuda(), Variable(target).cuda())
    print(ls)

    loss = IOULoss2d(num_classes=2)
    ls = loss(Variable(predict).cuda(), Variable(target).cuda())
    print(ls)
