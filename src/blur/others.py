# coding: utf-8

# 采用不同的分割方法，对结果进行分析。


from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from skimage import filters
import re
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pandas as pd

from src.eval import precision, recall, eval_pair, eval_pair_batch, f1

from sklearn.metrics import roc_curve, auc, precision_recall_curve

linestyles = ['-', ':', '-.', '--', ]
markers = [',', '*', '+', '.', '_', 'x', '^', 'o', 'v', '<', '>', '1', '2', '3', '4', 's', 'p', 'h', 'H', 'D', 'd', '|',
           ]

test_log_pt = '>>> pixel acc:\s+([\d]*.[\d]*), mean acc:\s+([\d]*.[\d]*) mIoU:\s+([\d]*.[\d]*) fIoU:\s+([\d]*.[\d]*)'


def extract_log(filename):
    lines = open(filename).readlines()
    ptest = re.compile(
        '>>> pixel acc:\s+([\d]*.[\d]*), mean acc:\s+([\d]*.[\d]*) miu:\s+([\d]*.[\d]*) fiu:\s+([\d]*.[\d]*)')
    ptrain = re.compile('epoch\s+\d+ step:\s+\d+ loss:\s+([\d]*.[\d]*) mean loss:\s+([\d]*.[\d]*)')

    train_loss = []
    test_loss = []
    for line in lines:
        mtrain = ptrain.match(line)
        if mtrain:
            train_loss.append(float(mtrain.groups(1)[1]))

        mtest = ptest.match(line)
        if mtest:
            test_loss.append([float(mtest.group(1)), float(mtest.group(2)), float(mtest.group(3)),
                              float(mtest.group(4))])

    train_loss = np.asarray(train_loss)
    test_loss = np.asarray(test_loss)

    return train_loss, test_loss


def extract_fc(filename):
    lines = open(filename).readlines()
    pt = re.compile("recall:\s+([\d]*.[\d]*)\s+precision:\s+([\d]*.[\d]*)\s+f1:\s+([\d]*.[\d]*)\s+acc:\s+([\d]*.[\d]*)")
    loss = []
    for line in lines:
        mtest = pt.match(line)
        if mtest:
            loss.append([float(mtest.group(1)), float(mtest.group(2)), float(mtest.group(3)), float(mtest.group(4))])

    loss = np.asarray(loss)
    return loss


def compare_st_msrcnn():
    st_train, st_test = extract_log('../../logs/resnet152-fcn2s-fc-st2.log')
    msrcnn_train, msrcnn_test = extract_log('../../logs/resnet152-fcn2s-fc-msrcnn.log')
    no_fc_train, no_fc_test = extract_log('../../logs/resnet152-fcn2s.log')

    label_list = ['MBlurNet-1', 'MBlurNet-2', 'BlurNet']
    plt.figure(figsize=(10, 5))
    plt.grid()

    cnt = 0
    for data, label in zip([st_train, msrcnn_train, no_fc_train], label_list):
        plt.plot(data, label=label, linestyle=linestyles[cnt])
        cnt += 1

    plt.legend()
    plt.title('Train Loss')
    plt.tight_layout()
    plt.savefig('fc-train_loss.png')

    labels = ['Pixel Accuracy', 'Mean Accuracy', 'Mean IoU', 'Frequency Mean IoU']
    for idx, label in zip(list(range(4)), labels):
        plt.figure()
        plt.grid()

        cnt = 0
        for data, label in zip([st_test, msrcnn_test, no_fc_test], label_list):
            # plt.plot(data[:, idx], label=label, linestyle=linestyles[cnt])
            plt.plot(data[:, idx], label=label)
            cnt += 1

        if idx + 1 == 4:
            plt.ylim([0.73, 0.88])
        plt.title(labels[idx])
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig('_'.join(labels[idx].split()) + '_fc.png')

    plt.show()


def compare_st_msrcnn_fc():
    st_test = extract_fc('logs/resnet152-fcn2s-fc-st.log')
    msrcnn_test = extract_fc('logs/resnet152-fcn2s-fc-msrcnn.log')
    label_list = ['FCN2s-ResNet152-ST', 'FCN2s-ResNet152-MSRCNN']

    labels = ['Recall', 'Precision', 'F1', 'Accuracy']
    for idx, label in zip(list(range(4)), labels):
        plt.figure()
        plt.grid()
        cnt = 0
        for data, lb in zip([st_test, msrcnn_test], label_list):
            plt.plot(data[:, idx], label=lb, color='black', linestyle=linestyles[cnt], marker=markers[cnt])
            cnt += 1

        plt.title(labels[idx])
        plt.legend(loc='lower right')
        plt.tight_layout()
        # plt.savefig('imgs/' + '_'.join(labels[idx].split()) + '_fc.png')

    plt.show()


def compare_by_metric(log_files, exp_label):
    log_data = [extract_log(log_file) for log_file in log_files]
    test_results = [e[1] for e in log_data]
    labels = ['Pixel Accuracy', 'Mean Accuracy', 'Mean IoU', 'Frequency Mean IoU']
    for idx, label in zip(list(range(len(exp_label))), labels):
        plt.figure()
        plt.grid()
        for cnt, (data, lb) in enumerate(zip(test_results, exp_label)):
            # plt.plot(data[:, idx], label=lb, linestyle=linestyles[cnt])
            plt.plot(data[:, idx], label=lb)

        plt.title(labels[idx])
        plt.legend()
        # lims = [[0.8, 0.92], [0.8, 0.90], [0.65, 0.825], [0.7, 0.88]]
        # plt.ylim(lims[idx])
        plt.tight_layout()
        # plt.savefig('_'.join(labels[idx].split()) + '.png')

    plt.show()


def compare_segnet_unet():
    unet_train, unet_test = extract_log('../../logs/unetv2.log')
    segnet_train, segnet_test = extract_log('../../logs/segnet.log')
    resnet_train, resnet_test = extract_log('../../logs/resnet152-fcn2s.log')
    dss_train, dss_test = extract_log('../../logs/deepsp.log')
    fcn_train, fcn_test = extract_log('../../logs/vgg16-fcn2s.log')

    print(resnet_train.shape, resnet_test.shape)
    print(segnet_train.shape, segnet_test.shape)
    print(unet_train.shape, unet_test.shape)
    print(fcn_train.shape, fcn_test.shape)

    test_results = [resnet_test, unet_test, segnet_test, dss_test, fcn_test]

    # test_results = test_results[:3]
    train_results = [resnet_train, unet_train, segnet_train, dss_train, fcn_train]
    # train_results = train_results[:3]

    label_list = ['DBNet', 'UNet', 'SegNet', 'DSS', 'FCN2s(VGG16)']
    # plt.figure(figsize=(10, 5))
    # plt.grid()
    # cnt = 0
    # for data, label in zip(train_results, label_list):
    #     plt.plot(data, label=label, linestyle=linestyles[cnt])
    #     cnt += 1
    #
    # plt.legend()
    # plt.title('Train Loss')
    # plt.tight_layout()
    # plt.savefig('unet-segnet-train_loss.png')
    # plt.show()

    labels = ['Pixel Accuracy', 'Mean Accuracy', 'Mean IoU', 'Frequency Mean IoU']
    for idx, label in zip(list(range(len(label_list))), labels):
        plt.figure()
        plt.grid()
        cnt = 0
        for data, lb in zip(test_results, label_list):
            # plt.plot(data[:, idx], label=lb, linestyle=linestyles[cnt])
            plt.plot(data[:, idx], label=lb)
            cnt += 1
        plt.title(labels[idx])
        plt.legend()
        lims = [[0.8, 0.92], [0.8, 0.90], [0.65, 0.825], [0.7, 0.88]]
        plt.ylim(lims[idx])
        plt.tight_layout()
        plt.savefig('_'.join(labels[idx].split()) + '.png')

    plt.show()


def load_all_imgs(base_path, func=None, flatten=True, concate=True):
    filenames = os.listdir(base_path)
    filenames = sorted(filenames)
    im_list = []
    for filename in filenames:
        im = cv2.imread(os.path.join(base_path, filename), cv2.IMREAD_GRAYSCALE)
        if func is not None:
            im = func(im)

        if flatten:
            im = im.reshape((-1,))

        im_list.append(im)
        # print(im.shape)

    if concate:
        return np.concatenate(im_list)

    return np.asarray(im_list)


# image_base = r'E:\Exp\blurcvpr
if os.name == 'nt':
    image_base = r'E:\Exp\blurcvpr'
else:
    image_base = r'/home/adam/Datasets/BlurDataset/result_val'

raw_base = os.path.join(image_base, 'image')
gt_base = os.path.join(image_base, 'gt')
test_image = 'motion0009{}.{}'

results_others = os.path.join(image_base, 'result_others')
results_shi = os.path.join(image_base, 'result_shi')

result_sm = r'E:\Blur\sm_results\result_sm'
result_raw = r'E:\Blur\sm_results\result_raw'

binary_sm = r'E:\Blur\binary_results\result_sm'
binary_raw = r'E:\Blur\binary_results\result_raw'


def sm_metrics():
    boarder = 7

    def gt_resize_crop(im):
        h, w = im.shape
        im = cv2.resize(im, (w // 2, h // 2))
        dh, dw = h // 2 - (2 * boarder + 1), w // 2 - (2 * boarder + 1)
        im = im[boarder:boarder + dh, boarder:boarder + dw]
        return im

    gt = load_all_imgs(gt_base, gt_resize_crop, flatten=False, concate=False)
    gt = gt // 255
    print(gt.shape)

    data = load_all_imgs(binary_sm, flatten=False, concate=False)
    data = data // 255
    print(data.shape)

    raw_data = load_all_imgs(binary_raw, flatten=False, concate=False)
    raw_data = raw_data // 255
    print(raw_data.shape)

    metrics = eval_pair_batch(data, gt)
    raw_metrics = eval_pair_batch(raw_data, gt)


def others_all():
    pass


def roc_pr_single(eval_dir):
    """
    ROC PR曲线
    :return:
    """

    boarder = 7

    def gt_resize_crop(im):
        h, w = im.shape
        im = cv2.resize(im, (w // 2, h // 2))
        dh, dw = h // 2 - (2 * boarder + 1), w // 2 - (2 * boarder + 1)
        im = im[boarder:boarder + dh, boarder:boarder + dw]
        return im

    gt = load_all_imgs(gt_base, gt_resize_crop)
    print(gt.shape)
    gt = gt // 255

    data = load_all_imgs(eval_dir)
    data = data / 255.0
    print(data.shape)

    _raw_data = load_all_imgs(result_raw)
    _raw_data = _raw_data / 255.0
    print(_raw_data.shape)

    print(f1(data, gt, all=True))
    print(f1(_raw_data, gt, all=True))

    plt.figure()
    p, r, _ = precision_recall_curve(gt, data)
    plt.plot(r, p, lw=2, label='result_sm')

    p, r, _ = precision_recall_curve(gt, _raw_data)
    plt.plot(r, p, lw=2, label='result_raw')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.legend()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.tight_layout()
    plt.savefig('sm_pr.png')
    plt.show()

    plt.figure()
    fpr, tpr, _ = roc_curve(gt, data)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label="ROC curve of result_sm (area = {:0.2f})".format(roc_auc))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)

    fpr, tpr, _ = roc_curve(gt, _raw_data)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label="ROC curve of result_raw (area = {:0.2f})".format(roc_auc))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('sm_roc.png')
    plt.show()


def my_roc_and_pr_curve():
    gt = load_all_imgs(gt_base)
    gt = gt // 255

    p = dict()
    r = dict()
    target_list = ['result_fft', 'result_lIoU', 'result_shi', 'result_su', 'result_zhang', 'result_ours']
    for dirname in target_list:
        data = load_all_imgs(os.path.join(image_base, dirname))
        data = data / 255.0
        print(dirname, data.shape)
        p[dirname], r[dirname], _ = precision_recall_curve(gt, data)
        del data
        # break

    label_list = ['result_chakrabarti', 'result_lIoU', 'result_shi', 'result_su', 'result_zhang', 'result_ours']
    colors = ['blue', 'black', 'coral', 'cyan', 'green', 'red']
    for idx, (target_name, dirname, color) in enumerate(zip(target_list, label_list, colors)):
        # plt.plot(r[dirname], p[dirname], lw=1, label=dirname[7:], linestyle=linestyles[idx % len(linestyles)]
        #          , marker=markers[idx % len(markers)], markevery=0.1)
        plt.plot(r[target_name], p[target_name], lw=1, label=dirname[7:])
        # break

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.legend()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.tight_layout()
    plt.savefig('others-pr.png')
    # plt.show()

    # roc
    plt.figure()
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    # target_list = ['result_lIoU', 'result_su', 'result_shi', 'result_chakrabarti']
    for dirname in target_list:
        data = load_all_imgs(os.path.join(image_base, dirname))
        print(dirname, data.shape)
        fpr[dirname], tpr[dirname], _ = roc_curve(gt, data)
        roc_auc[dirname] = auc(fpr[dirname], tpr[dirname])
        del data

    for idx, target_name, label_name in zip(range(len(target_list)), target_list, label_list):
        # plt.plot(fpr[ke], tpr[ke], lw=1,
        #          label='ROC of {0}(area = {1:0.2f})'
        #                ''.format(ke[7:], roc_auc[ke]), linestyle=linestyles[idx % len(linestyles)]
        #          , marker=markers[idx % len(markers)], markevery=0.1)

        plt.plot(fpr[target_name], tpr[target_name], lw=1,
                 label='ROC of {0}(area = {1:0.2f})'
                       ''.format(label_name[7:], roc_auc[target_name]), color=colors[idx])

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.legend(loc="lower right")
    plt.legend()
    plt.tight_layout()
    plt.savefig('others-roc.png')
    plt.show()


def copy():
    import shutil
    from_dir = '/home/adam/Datasets/BlurDataset/gt'
    dest_dir = '/home/adam/Datasets/BlurDataset/result_val/gt'
    reference_dir = '/home/adam/Datasets/BlurDataset/result_val/result_ours'
    for filename in os.listdir(reference_dir):
        filename = filename[:-8] + '.png'
        shutil.copy(os.path.join(from_dir, filename), os.path.join(dest_dir, filename))


from skimage.filters import threshold_otsu
from tqdm import tqdm


def eval_sm(result_list, gt_list, up_sample=False):
    """
    评估sm方法的结果，两个列表的长度应该保持一致
    :param result_list: list
    :param gt_list: list
    :return:
    """

    def align(result, gt):
        h, w = result.shape
        gh, gw = gt.shape
        hs, ws = abs(h - gh) // 2, abs(w - gw) // 2
        gt = gt[hs:hs + h, ws:ws + w]
        return result, gt

    assert len(result_list) == len(gt_list)
    pa_list, ma_list, mIoU_list, fIoU_list = [], [], [], []
    for path_result, path_gt in tqdm(zip(result_list, gt_list)):
        result, gt = cv2.imread(path_result, cv2.IMREAD_GRAYSCALE), cv2.imread(path_gt, cv2.IMREAD_GRAYSCALE)
        if up_sample:
            gt = cv2.resize(gt, (gt.shape[0] // 2, gt.shape[1] // 2))
        result_binary = result > threshold_otsu(result)
        result = result_binary.astype(np.uint8)
        gt = gt // 255
        # print(result.shape, gt.shape)
        result, gt = align(result, gt)
        pa, ma, mIoU, fIoU = eval_pair(result, gt)
        pa_list.append(pa)
        ma_list.append(ma)
        mIoU_list.append(mIoU)
        fIoU_list.append(fIoU)

    df = pd.DataFrame({'pa': pa_list, 'ma': ma_list, 'mIoU': mIoU_list, 'fIoU': fIoU_list})
    return df


def eval_all_sm(base_dir, gt_dir):
    """
    评估sm中所有的方法
    :param base_dir:
    :param gt_dir:
    :return:
    """
    gt_list = [os.path.join(gt_dir, gt) for gt in sorted(os.listdir(gt_dir))]
    results = os.listdir(base_dir)

    upsample = True
    with open('results.txt', 'w+') as final_results:
        for result in results:
            result_list = [os.path.join(base_dir, result, f) for f in
                           sorted(os.listdir(os.path.join(base_dir, result)))]
            if result == 'result_raw':
                upsample = False

            df_metrics = eval_sm(result_list, gt_list, up_sample=upsample)
            df_metrics.to_csv(result + '.csv', index=False)
            mean = df_metrics.mean()
            print(mean)
            final_results.write(mean)
            final_results.write('\n')


others_base = r'E:\Exp\blurcvpr'
result_others_folder = os.path.join(others_base)
import glob
from skimage.io import imsave


def ostu_binary():
    methods = ['result_fft', 'result_lIoU', 'result_shi', 'result_su', 'result_zhang']
    for method in methods[-1:]:
        result_folder = os.path.join(others_base, 'binary_{}'.format(method))
        os.makedirs(result_folder, exist_ok=True)
        filepaths = glob.glob(os.path.join(others_base, method) + "/*")
        for filepath in filepaths:
            # print(filepath)
            im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            th = 0
            try:
                th = filters.threshold_otsu(im)
            except ValueError:
                print('eror ....')

            binary = (im > th).astype('uint8') * 255
            result_path = filepath.replace(method, 'binary_' + method)
            print(result_path)
            imsave(result_path, binary)


def all_binary_metrics():
    gt = load_all_imgs(gt_base, flatten=False, concate=False)
    gt = gt // 255
    print(gt.shape)

    methods = ['result_fft', 'result_lIoU', 'result_shi', 'result_su', 'result_zhang']
    for method in methods[-1:]:
        binary_folder = r'E:\Exp\blurcvpr\binary_{}'.format(method)
        data = load_all_imgs(binary_folder, flatten=False, concate=False)
        data = data // 255
        print(data.shape)

        metrics = eval_pair_batch(data, gt)
        del data


def compare_sample_weight():
    log_files = ['../../logs/resnet152-fcn2s-pweight-{}-lr.log'.format(i) for i in (3, 6, 9)]
    log_files.insert(0, '../../logs/resnet152-fcn2s.log')
    exp_label = ['DBNet({}:1)'.format(i) for i in (1, 3, 6, 9)]
    compare_by_metric(log_files, exp_label)


if __name__ == '__main__':
    compare_sample_weight()
    # eval_all_sm(r'E:\Blur\sm_results', r'E:\Blur\blurdetect_cvpr14\BlurDatasetGT\gt')
    # compare_segnet_unet()
    # my_roc_and_pr_curve()
    # roc_pr_single(result_sm)
    # copy()
    # compare_st_msrcnn()
    # compare_st_msrcnn_fc()
    # ostu_binary()
    # all_binary_metrics()
    # sm_metrics()
