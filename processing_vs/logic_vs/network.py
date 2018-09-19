#!/usr/bin/env python
# -*- coding: utf-8 -*-

import caffe
import numpy as np
from joblib import Parallel, delayed, cpu_count


def apply_infer(patch, net):
    _patch = patch[1][..., ::-1]
    # _patch -= _patch.mean(axis=0)
    _patch = _patch.transpose((2, 0, 1))

    net.blobs['data'].reshape(1, *_patch.shape)
    net.blobs['data'].data[...] = _patch
    net.forward()

    out = net.blobs['score'].data[0].argmax(axis=0)
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
        if use_gpu:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

        self.net = caffe.Net(deploy_fpath, model_fpath, caffe.TEST)

    def infer(self, patches):

        # from PIL import Image
        # for i, patch in enumerate(patches[:5]):
        #     _patch = patch[1]
        #     result = Image.fromarray(_patch)
        #     result.save('/home/delmonte/Desktop/png/{}.png'.format(i))

        num_cores = cpu_count()
        with Parallel(n_jobs=num_cores, backend='threading') as parallel:
            seg = parallel(delayed(apply_infer)(patch, self.net) for patch in patches)

        # seg = []
        # for patch in patches:
        #     seg.append(apply_infer(patch, self.net))

        return seg
