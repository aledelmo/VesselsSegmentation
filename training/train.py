#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os.path
import sys
from time import time

from functions import solve


def main():
    solver_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'files', 'solver.prototxt')
    base_deploy = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'files', 'VGG16',
                               'VGG_ILSVRC_16_layers_deploy.prototxt')
    base_model = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'files', 'VGG16',
                              'VGG_ILSVRC_16_layers.caffemodel')
    validation_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'files', 'dataset', 'val')

    solve(solver_path, base_deploy, base_model, validation_path)


if __name__ == '__main__':
    t0 = time()
    main()
    print('Execution time: {} s'.format(round((time() - t0), 2)))

    sys.exit(0)
