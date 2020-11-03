#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import os
import sys

import nibabel as nib
import numpy as np
from PIL import Image
from skimage.measure import label, regionprops


def load_nii(fname):
    """
    NIfTI images loading
    :param fname: filename
    :return: data array, affine matrix
    """
    img = nib.load(fname)
    canonical_img = nib.as_closest_canonical(img)
    return canonical_img.get_fdata(), canonical_img.affine, canonical_img.header.get_zooms()


def get_patches(ref_path, lab_path, dir_path, legacy):
    ref, affine, _ = load_nii(ref_path)
    lab, check_aff, _ = load_nii(lab_path)

    iter_name = len([name for name in os.listdir(dir_path) if name.endswith('.png')]) + 1

    if not np.all(affine == check_aff):
        sys.exit(1)

    if legacy:
        a = 17
        v = 16
    else:
        a = 43
        v = 44

    lab[(lab != a) & (lab != v)] = 0
    lab[lab == a] = 1
    lab[lab == v] = 2

    ref = (255 * (ref / np.amax(ref))).astype(np.uint8)
    for s in range(ref.shape[-1]):
        current = lab[:, :, s]
        regions = regionprops(label(current))

        for c in regions:
            x, y = [int(_) for _ in c.centroid]
            patch = ref[x - 15:x + 16, y - 15:y + 16, s - 1:s + 2]
            im = Image.fromarray(patch)
            path = os.path.join(dir_path, '{}.png'.format(iter_name))
            im.save(path)

            patch = (lab[x - 15:x + 16, y - 15:y + 16, s]).astype(np.uint8)
            im = Image.fromarray(patch)
            path = os.path.join(dir_path, '{}.pgm'.format(iter_name))
            im.save(path)

            iter_name += 1
