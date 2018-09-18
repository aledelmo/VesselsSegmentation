#!/usr/bin/env python
# -*- coding: utf-8 -*-

import caffe
import numpy as np
from joblib import Parallel, delayed, cpu_count


def par_infer(patch, net):
    _patch = (patch[1][:, :, ::-1]).astype(np.uint8)
    # _patch -= _patch.mean(axis=-1)
    _patch = _patch.transpose((2, 0, 1))

    net.blobs['data'].reshape(1, *_patch.shape)
    net.blobs['data'].data[...] = _patch
    net.forward()

    out = net.blobs['score'].data[0].argmax(axis=0)
    # out = tuple(patch[0], out)
    return patch[0], out


class Cnn:
    def __init__(self):
        caffe.set_device(0)
        self.net = None

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)

    def __str__(self):
        return "{}()".format('Convolutional Neural Network')

    def load_net(self, deploy_fpath, model_fpath, use_gpu):
        use_gpu = False
        if use_gpu:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        self.net = caffe.Net(deploy_fpath, model_fpath, caffe.TEST)

    def infer(self, patches):
        # seg = []

        num_cores = cpu_count()
        with Parallel(n_jobs=num_cores, backend='threading') as parallel:
            seg = parallel(delayed(par_infer)(patch, self.net) for patch in patches)
        # for patch in patches:
        #     _patch = (patch[1][:, :, ::-1]).astype(np.uint8)
        #     # _patch -= _patch.mean(axis=-1)
        #     _patch = _patch.transpose((2, 0, 1))
        #
        #     self.net.blobs['data'].reshape(1, *_patch.shape)
        #     self.net.blobs['data'].data[...] = _patch
        #     self.net.forward()
        #
        #     out = self.net.blobs['score'].data[0].argmax(axis=0)
        #     seg.append((patch[0], out))
        return seg
