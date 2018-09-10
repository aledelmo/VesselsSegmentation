#!/usr/bin/env python
# -*- coding: utf-8 -*-

from processing_vs.logic_vs import *


def proc(volume, seeds, vessel_type, output, from_plugin):
    data, affine, _ = load_nii(volume)
    points = load_pkl(seeds)

    skeleton = Tree(points, affine)
