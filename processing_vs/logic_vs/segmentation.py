#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def reconstruct(patches, reference, label):
    seg = np.zeros(reference.shape, dtype=np.uint8)
    for patch in patches:
        center = patch[0]
        seg[center[0] - 15:center[0] + 16, center[1] - 15:center[1] + 16, center[2]] = np.logical_or(
            seg[center[0] - 15:center[0] + 16, center[1] - 15:center[1] + 16, center[2]], patch[1]) * label

    return seg
