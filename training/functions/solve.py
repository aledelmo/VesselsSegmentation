#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import os
import sys
from builtins import range
from contextlib import contextmanager

import caffe
import numpy as np

from .score import seg_tests
from .surgery import transplant, interp

sys.path.append('/workspace/functions')


@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def solve(solver_path, base_deploy, base_model, validation_path):
    # caffe.set_device(0)
    # caffe.set_mode_gpu()
    caffe.set_mode_cpu()

    print('Loading solver')
    solver = caffe.SGDSolver(solver_path)

    print('Net Surgery')
    base_net = caffe.Net(base_deploy, base_model, caffe.TEST)
    transplant(solver.net, base_net)
    del base_net
    interp_layers = [k for k in list(solver.net.params.keys()) if 'up' in k]
    interp(solver.net, interp_layers)
    # val = np.loadtxt(validation_path, dtype=str)
    val = []
    with cd(validation_path):
        for file in glob.glob("*.png"):
            val.append(os.path.splitext(file)[0])
    val = np.sort(np.array(val))

    print('Training Started')
    for x in range(0, 20):
        solver.step(5000)
        seg_tests(solver, False, val, layer='score')
