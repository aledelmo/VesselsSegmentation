#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import os
import sys
from builtins import range

import nibabel as nib
import numpy as np
from PIL import Image
from skimage.measure import label, regionprops


# from scipy.ndimage import label
# from scipy.ndimage.measurements import center_of_mass


def load_nii(fname):
    """
    NIfTI images loading
    :param fname: filename
    :return: data array, affine matrix
    """
    img = nib.load(fname)
    canonical_img = nib.as_closest_canonical(img)
    canonical_data = canonical_img.get_data()
    canonical_affine = canonical_img.affine
    canonical_vox = canonical_img.header.get_zooms()
    return canonical_data, canonical_affine, canonical_vox


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
        # regions = label(current, structure=np.ones((3, 3)))[0]
        # centroids = center_of_mass(current, regions)
        # if len(centroids) == 2:
        #     centroids = [centroids]

        # if not np.isnan(centroids[0]).any():
        for c in regions:
        # for c in centroids:
            x, y = [int(_) for _ in c.centroid]
            # x, y = [int(_) for _ in c]
            patch = ref[x - 15:x + 16, y - 15:y + 16, s - 1:s + 2]
            im = Image.fromarray(patch)
            path = os.path.join(dir_path, '{}.png'.format(iter_name))
            im.save(path)

            patch = (lab[x - 15:x + 16, y - 15:y + 16, s]).astype(np.uint8)
            im = Image.fromarray(patch)
            path = os.path.join(dir_path, '{}.pgm'.format(iter_name))
            im.save(path)

            iter_name += 1
