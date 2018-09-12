#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from nibabel.affines import apply_affine
from .distance_measure import distance, bresenhamline
from anytree import Node, RenderTree, PreOrderIter

from builtins import int


class Tree:
    def __init__(self, points, affine):
        self.points = apply_affine(np.linalg.inv(affine), np.array(points)).astype(np.int16)
        self.points = self.points[self.points[:, 2].argsort()][::-1]
        self.affine = affine

        self.build_tree()

    def __repr__(self):
        return "{}({},{})".format(self.__class__.__name__, self.points, self.affine)

    def __str__(self):
        return "{}({},{})".format(self.__class__.__name__, 'Points_list', 'Affine')

    def build_tree(self):
        self.tree = Node('root', voxel=self.points[0])
        for point in self.points[1:]:
            min_dist = np.inf
            best_node = None
            for node in PreOrderIter(self.tree):
                if len(node.children) < 2 and int(node.voxel[2]) > int(point[2]):
                    current_dist = distance(point, node.voxel, node.parent)
                    if current_dist < min_dist:
                        best_node = node
                        min_dist = current_dist

            Node('leaf', voxel=point, parent=best_node, dist=min_dist)

        print(RenderTree(self.tree))

    def get_label_mask(self, ref_img, label_number):
        mask = np.zeros(ref_img.shape, dtype=np.uint16)
        for node in PreOrderIter(self.tree):
            for child in node.children:
                # mask[self.points[:, 0], self.points[:, 1], self.points[:, 2]] = label_number
                idx = bresenhamline(node.voxel[:, np.newaxis].T, child.voxel[:, np.newaxis].T, max_iter=-1).astype(
                    np.int16)
                mask[idx[:, 0], idx[:, 1], idx[:, 2]] = label_number
        return mask

    def get_patches(self, ref_img):
        patches = []
        for node in PreOrderIter(self.tree):
            for child in node.children:
                idx = bresenhamline(node.voxel[:, np.newaxis].T, child.voxel[:, np.newaxis].T, max_iter=-1).astype(
                    np.int16)
                for i in idx:
                    patch = ref_img[i[0] - 15:i[0] + 16, i[1] - 15:i[1] + 16, i[2] - 1:i[2] + 2]
                    patches.append((i, patch))
        return patches
