#!/usr/bin/env python
# -*- coding: utf-8 -*-

from processing_vs.logic_vs import *
import os.path


def proc(volume, seeds, vessel_type, output):
    data, affine, _ = load_nii(volume)
    points = load_pkl(seeds)
    if vessel_type.lower() == 'artery':
        label_number = 17
    else:
        label_number = 16

    cnn = Cnn()
    hierarchy = Tree(points, affine)

    # mask_skeleton = hierarchy.get_label_mask(data, label_number)
    # save_nii(mask_skeleton, affine, '/Users/imag2/Desktop/skeleton-label.nii.gz')

    deploy_fpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'cnn_models', 'deploy.prototxt')
    model_fpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'cnn_models',
                               'snapshot_iter_135000.caffemodel')

    cnn.load_net(deploy_fpath, model_fpath, True)
    out_patches = cnn.infer(hierarchy.get_patches(data))

    output_data = reconstruct(out_patches, data, label_number)

    output_data[output_data == 1] = 17
    output_data[output_data == 2] = 16

    save_nii(output_data, affine, output)
