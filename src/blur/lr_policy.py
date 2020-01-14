# coding: utf-8
from __future__ import print_function
from torch.optim import lr_scheduler


# The learning rate decay policy. The currently implemented learning rate
# policies are as follows:
#    - fixed: always return base_lr.
#    - step: return base_lr * gamma ^ (floor(iter / step))
#    - exp: return base_lr * gamma ^ iter
#    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
#    - multistep: similar to step but it allows non uniform steps defined by
#      stepvalue
#    - poly: the effective learning rate follows a polynomial decay, to be
#      zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)
#    - sigmoid: the effective learning rate follows a sigmod decay
#      return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
#
# where base_lr, max_iter, gamma, step, stepvalue and power are defined
# in the solver parameter protocol buffer, and iter is the current iteration.


def ser_opt_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def ploy_lr_policy(optimizer, init_lr, iter, lr_decay_iter=8214 * 10, max_iter=8214 * 50, power=0.9):
    if not (iter % lr_decay_iter == 0 and iter < max_iter):
        return

    lr = init_lr * (1 - iter / max_iter) ** power
    print('lr is set to: ', lr)
    set(optimizer, lr)

    return lr


def exp_lr_policy(optimizer, init_lr, n_epoch, gamma=0.1, epoch_decay=10):
    lr = init_lr * (gamma ** (n_epoch / epoch_decay))
    print('lr is set to: ', lr)
    set(optimizer, lr)

    return lr
