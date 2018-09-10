#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Tree:
    def __init__(self, points, affine):
        self.points = points
        self.affine = affine

    def __repr__(self):
        return "{}({},{})".format(self.__class__.__name__, self.points, self.affine)

    def __str__(self):
        return "{}({},{})".format(self.__class__.__name__, 'Points_list', 'Affine')
