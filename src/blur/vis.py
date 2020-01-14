# coding: utf-8
from __future__ import print_function

from argparse import ArgumentParser
import re
import matplotlib.pyplot as plt
import numpy as np


def log_origin(args):
    log_files = args.log_files.split(',')
    for log_file in log_files:
        mean_loss_epoch = []
        with open(log_file) as log:
            for line in log.readlines():
                pt = re.compile(r'epoch: (\d{3})>> mean loss: ([\d]*.[\d]*)')
                match_result = pt.match(line)
                if match_result:
                    mean_loss_epoch.append(float(match_result.groups(2)[1]))

        plt.plot(np.arange(1, len(mean_loss_epoch) + 1), mean_loss_epoch, label=log_file)

    plt.title('mean loss')
    plt.grid()
    plt.legend()
    plt.show()


def log_new(args):
    lines = open(args.log_file).readlines()
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
    plt.figure()
    plt.grid()
    length = range(len(test_loss))
    legends = ['pixel acc', 'mean acc', 'miu', 'fiu']
    colors = ['black', 'blue', 'red', 'yellow']
    for i, label in enumerate(legends):
        plt.plot(length, test_loss[:, i], label=label, color=colors[i])

    print(np.mean(test_loss[-20:-15, :], axis=0))
    # plt.plot(length, test_loss[:, 0], length, test_loss[:, 1], length, test_loss[:, 2], length, test_loss[:, 3])
    plt.legend()
    plt.title(args.log_file)
    plt.ylim([0.5, 1.0])

    # train
    plt.figure()
    plt.grid()
    iters_per_epoch = len(train_loss) // len(test_loss)
    plt.xticks(np.asarray(range(len(test_loss))) * iters_per_epoch,
               ['epoch {}'.format(i) for i in range(len(test_loss))], rotation=60)
    plt.plot(train_loss)
    plt.show()


def log_fc(args):
    lines = open(args.log_file).readlines()
    pt = re.compile("recall:\s+([\d]*.[\d]*)\s+precision:\s+([\d]*.[\d]*)\s+f1:\s+([\d]*.[\d]*)\s+acc:\s+([\d]*.[\d]*)")
    loss = []
    for line in lines:
        mtest = pt.match(line)
        if mtest:
            loss.append([float(mtest.group(1)), float(mtest.group(2)), float(mtest.group(3)), float(mtest.group(4))])

    loss = np.asarray(loss)
    print(np.max(loss, 0))
    plt.figure()
    plt.grid()
    length = range(len(loss))
    print(loss.shape)
    target_list = ['recall', 'precision', 'f1', 'accuracy']
    for idx, target in enumerate(target_list):
        plt.plot(loss[:, idx], label=target)
    # plt.plot(length, loss[:, 0], length, loss[:, 1], length, loss[:, 2], length, loss[:, 3])
    plt.legend()
    plt.show()


def diff(args):
    pass


def main(args):
    if args.type == 'new':
        log_new(args)

    elif args.type == 'fc':
        log_fc(args)

    else:
        log_origin(args)


if __name__ == '__main__':
    ps = ArgumentParser()
    ps.add_argument("--log-files")
    ps.add_argument('--log-file')
    ps.add_argument('--type')
    main(ps.parse_args())


# avg
# [0.86369325,0.85598332,0.74491477,0.8127626 ]
# [0.93033495,0.90971139,0.83843323,0.88593397]
# [0.91183559,0.89386991,0.81201327,0.86648541]
# [0.90812105, 0.89180203, 0.80726041, 0.86154351]

# weight
# [0.91183559 0.89386991 0.81201327 0.86648541]
# [0.90204591 0.87992908 0.7949282  0.85335724]
# [0.89530693 0.87218793 0.78425968 0.84457799]
# [0.88833782 0.86970866 0.77529886 0.83479205]

