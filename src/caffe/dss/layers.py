# coding: utf-8
from __future__ import print_function
from __future__ import division
import sys

caffe_root = '/home/adam/Gitss/caffe_dss/'
sys.path.insert(0, caffe_root + 'python')

import caffe
import time
import numpy as np
import cv2

from src.eval import eval_pair_batch


class EvalLayer(caffe.Layer):

    def setup(self, bottom, top):
        if len(bottom) != 2:
            raise Exception("Need to define two tops: data and label.")

        self.pp = eval(self.param_str)
        self.iter = self.pp['iter']
        self.cur = 0

        self.results = np.zeros((self.iter, 4), dtype=np.float32)

    def forward(self, bottom, top):
        eval_result = eval_pair_batch(bottom[0].data, bottom[1].data, print_result=False)
        # for idx, img in enumerate(bottom[0].data):
        #     cv2.imwrite('debug/{}_{}_pred.png'.format(self.cur, idx), (np.squeeze(img)*255).astype(np.uint8))

        # for idx, img in enumerate(bottom[1].data):
        #     cv2.imwrite('debug/{}_{}_gt.png'.format(self.cur, idx), (np.squeeze(img)*255).astype(np.uint8))

        self.results[self.cur, ...] = np.asarray(eval_result)
        self.cur += 1
        print(self.cur, eval_result)
        
        if self.cur == self.iter:
            top[0].data[...] = self.results.mean(axis=0)
            print(self.results.mean(axis=0))
            self.cur = 0

    def reshape(self, bottom, top):
        top[0].reshape(4)

    def backward(self, top, propagate_down, bottom):
        pass


if __name__ == '__main__':
    pass
