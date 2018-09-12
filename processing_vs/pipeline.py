#!/usr/bin/env python
# -*- coding: utf-8 -*-

from processing_vs.logic_vs import *
import nibabel as nib
import numpy as np


def proc(volume, seeds, vessel_type, output, from_plugin):
    data, affine, vox_dim = load_nii(volume)
    points = load_pkl(seeds)

    hierarchy = Tree(points, affine, vox_dim)

    if vessel_type.lower() == 'artery':
        label_number = 17
    else:
        label_number = 16

    # output_data = hierarchy.get_label_mask(data, label_number)
    output_data = hierarchy.get_patches(data)

    # img = nib.Nifti1Image(output_data.astype(np.int16), affine)
    # nib.save(img, output)
