#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from skimage.measure import label


def reconstruct(patches, reference):
    seg = np.zeros(reference.shape, dtype=np.uint8)
    for patch in patches:
        center = patch[0]
        conn = label(patch[1], neighbors=8)
        current = seg[center[0] - 15:center[0] + 16, center[1] - 15:center[1] + 16, center[2]]
        tokeep = conn[15, 15]
        if tokeep != 0:
            for index, value in np.ndenumerate(patch[1]):
                if conn[index] == tokeep:
                    patch[1][index] = 1
                elif conn[index] != tokeep and value != 0:
                    patch[1][index] = 3
            for index, value in np.ndenumerate(current):
                if patch[1][index] == 1:
                    current[index] = 1
                elif patch[1][index] == 3 and value == 0:
                    current[index] = 3
        else:
            unique, counts = np.unique(conn, return_counts=True)
            unique_dict = dict(zip(unique, counts))
            del unique_dict[0]
            if unique_dict:
                largest = max(unique_dict, key=lambda key: unique_dict[key])
                for index, value in np.ndenumerate(patch[1]):
                    if conn[index] == largest:
                        patch[1][index] = 3
            patch[1][15, 15] = 1
            for index, value in np.ndenumerate(current):
                if patch[1][index] == 1:
                    current[index] = 1
                elif patch[1][index] == 3 and value == 0:
                    current[index] = 3

        seg[center[0] - 15:center[0] + 16, center[1] - 15:center[1] + 16, center[2]] = current

    return seg


def improve_seg(img, artery, vein):
    to_process = np.transpose(np.where(img == 3)).tolist()
    changed = True
    while to_process and changed is True:
        changed = False
        for i, p in enumerate(to_process):
            neigh = img[p[0] - 1:p[0] + 2, p[1] - 1:p[1] + 2, p[2] - 1:p[2] + 2]
            if vein in neigh:
                img[p[0], p[1], p[2]] = vein
                changed = True
                del to_process[i]
            elif artery in neigh:
                img[p[0], p[1], p[2]] = artery
                changed = True
                del to_process[i]
    img[img == 3] = 0
    return img
