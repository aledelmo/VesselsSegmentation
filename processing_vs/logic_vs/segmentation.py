#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from skimage.measure import label


def reconstruct(patches, reference):
    seg = np.zeros(reference.shape, dtype=np.uint8)
    for patch in patches:
        center = patch[0]
        #########################
        # current = seg[center[0] - 15:center[0] + 16, center[1] - 15:center[1] + 16, center[2]]
        # for index, value in np.ndenumerate(current):
        #     if value == 0:
        #         current[index] = patch[1][index]
        #     elif value != 0 and value != patch[1][index]:
        #         current[index] = 3
        # seg[center[0] - 15:center[0] + 16, center[1] - 15:center[1] + 16, center[2]] = current
        ########################3
        # seg[center[0] - 15:center[0] + 16, center[1] - 15:center[1] + 16, center[2]] = np.where(
        #     (seg[center[0] - 15:center[0] + 16, center[1] - 15:center[1] + 16, center[2]] - patch[1]) == 0,
        #     seg[center[0] - 15:center[0] + 16, center[1] - 15:center[1] + 16, center[2]], patch[1])
        ###########################3
        # to_update = None
        # conn, n_conn = label(patch[1], neighbors=8, return_num=True)
        # for n in range(n_conn):
        #     i, j = np.where(conn == n)
        #     for _i, _j in zip(i, j):
        #         if _i == 15 and _j == 15:
        #             to_update = n
        # print(to_update)
        # final_patch = np.zeros(patch[1].shape)
        # final_patch[np.where(conn == to_update)] = 1
        ################################
        conn = label(patch[1], neighbors=8)
        tokeep = conn[15, 15]
        if tokeep != 0:
            for index, value in np.ndenumerate(patch[1]):
                if conn[index] == tokeep and tokeep != 0:
                    patch[1][index] = 1
                elif conn[index] != tokeep and value != 0:
                    patch[1][index] = 3
            current = seg[center[0] - 15:center[0] + 16, center[1] - 15:center[1] + 16, center[2]]
            for index, value in np.ndenumerate(current):
                if patch[1][index] == 1:
                    current[index] = 1
                elif patch[1][index] == 3 and value == 0:
                    current[index] = 3

            seg[center[0] - 15:center[0] + 16, center[1] - 15:center[1] + 16, center[2]] = current
        else:
            seg[center[0], center[1], center[2]] = 1
        #
        # seg[center[0] - 15:center[0] + 16, center[1] - 15:center[1] + 16, center[2]] = np.logical_or(
        #     seg[center[0] - 15:center[0] + 16, center[1] - 15:center[1] + 16, center[2]], final_patch)
        ###########################
        # seg[center[0] - 15:center[0] + 16, center[1] - 15:center[1] + 16, center[2]] = np.logical_or(
        #     seg[center[0] - 15:center[0] + 16, center[1] - 15:center[1] + 16, center[2]], patch[1])
    return seg
