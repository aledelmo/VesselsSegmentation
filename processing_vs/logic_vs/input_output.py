#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np
import pickle


def load_nii(fname):
    """
    NIfTI images loading
    :param fname: filename
    :return: data array, affine matrix
    """
    img = nib.load(fname)
    canonical_img = nib.as_closest_canonical(img)
    return canonical_img.get_fdata(), canonical_img.affine, canonical_img.header.get_zooms()


def save_nii(data, affine, fname):
    img = nib.Nifti1Image(data.astype(np.int16), affine)
    nib.save(img, fname)


def load_pkl(fname):
    with open(fname, 'rb') as handle:
        body = pickle.load(handle)
    return body
