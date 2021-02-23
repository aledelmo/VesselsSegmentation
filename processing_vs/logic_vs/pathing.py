#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np


def distance(current, candidate, parent, alpha=200):
    if parent is not None:
        parent = parent.voxel

        _a = np.linalg.norm(parent - candidate)
        _b = np.linalg.norm(parent - current)
        _c = np.linalg.norm(candidate - current)

        s = (_a + _b + _c) / 2.

        r = _a * _b * _c / 4 / np.sqrt(s * (s - _a) * (s - _b) * (s - _c))

    else:
        r = np.inf

    curv = 1. / r

    dist = np.linalg.norm(current - candidate)

    return dist + alpha * curv


def _bresenhamline_nslope(slope):
    scale = np.amax(np.abs(slope), axis=1).reshape(-1, 1)
    zeroslope = (scale == 0).all(1)
    scale[zeroslope] = np.ones(1)
    normalizedslope = np.array(slope, dtype=np.double) / scale
    normalizedslope[zeroslope] = np.zeros(slope[0].shape)
    return normalizedslope


def _bresenhamlines(start, end, max_iter):
    if max_iter == -1:
        max_iter = np.amax(np.amax(np.abs(end - start), axis=1))
    npts, dim = start.shape
    nslope = _bresenhamline_nslope(end - start)

    stepseq = np.arange(1, max_iter + 1)
    stepmat = np.tile(stepseq, (dim, 1)).T

    bline = start[:, np.newaxis, :] + nslope[:, np.newaxis, :] * stepmat

    return np.array(np.rint(bline), dtype=start.dtype)


def bresenhamline(start, end, max_iter=5):
    return _bresenhamlines(start, end, max_iter).reshape(-1, start.shape[-1])
