#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path

import numpy as np

from processing_vs.logic_vs import *


def proc(volume, arteries, veins, output):
    data, affine, _ = load_nii(volume)
    points = []
    if arteries is not None:
        a = load_pkl(arteries)
        if a is not None:
            points.append(a)
    if veins is not None:
        v = load_pkl(veins)
        if v is not None:
            points.append(v)

    use_gpu = False
    cnn = Cnn(use_gpu)
    deploy_fpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'cnn_models', 'deploy.prototxt')
    # model_fpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'cnn_models',
    #                            'snapshot_iter_135000.caffemodel')
    model_fpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'cnn_models',
                               'benchmark.caffemodel')
    output_data = []
    vein_label = 16
    art_label = 17
    for i, init in enumerate(points):
        hierarchy = Tree(init, affine)

        # mask_skeleton = hierarchy.get_label_mask(data, label_number)
        # save_nii(mask_skeleton, affine, '/Users/imag2/Desktop/skeleton-label.nii.gz')

        cnn.load_net(deploy_fpath, model_fpath)
        out_patches = cnn.infer(hierarchy.get_patches(data))

        seg = reconstruct(out_patches, data)
        if i == 1:
            seg[seg == 1] = vein_label
        else:
            if len(points) > 1 and i == 0:
                seg[seg == 1] = art_label
            else:
                if arteries is not None:
                    seg[seg == 1] = art_label
                else:
                    seg[seg == 1] = vein_label

        output_data.append(seg)

    if len(points) > 1:
        combined = np.maximum(output_data[0], output_data[1])
    else:
        combined = output_data[0]

    final_seg = improve_seg(combined, art_label, vein_label)

    save_nii(final_seg, affine, output)
