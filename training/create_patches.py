#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os.path
import sys
from time import time

from patches import get_patches

__author__ = 'Alessandro Delmonte'
__email__ = 'delmonte.ale92@gmail.com'


def main():
    """
    Call to parser and pipeline

    :rtype: void
    """
    ref_path, lab_path, dir_path, legacy = setup()
    get_patches(ref_path, lab_path, dir_path, legacy)


def setup():
    """
    Command line parser

    :rtype: list of strings
    :return: arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('Input_Ref_Volume', help='Name of the input reference volume', type=check_nii)
    parser.add_argument('Input_Lab_Volume', help='Name of the input label volume', type=check_nii)
    parser.add_argument('Output_Dir', help='Name of the output directory', type=check_dir)
    parser.add_argument('-l', '--legacy', help='Use legacy colorLUT.', action='store_true')

    args = parser.parse_args()

    return args.Input_Ref_Volume, args.Input_Lab_Volume, args.Output_Dir, args.legacy


def check_nii(value):
    """
    Verify that the file has a valid NIfTI-1 extension

    :param value: command line inserted string
    :return: file absolute path
    :rtype: string
    """
    if value.endswith('.nii') or value.endswith('.nii.gz') and os.path.isfile(os.path.abspath(value)):
        return os.path.abspath(value)
    else:
        raise argparse.ArgumentTypeError("Invalid output extension (file format supported: nii, nii.gz): %s" % value)


def check_dir(value):
    if os.path.isdir(value):
        return os.path.abspath(value)
    else:
        raise argparse.ArgumentTypeError("Invalid output directory: %s" % value)


if __name__ == '__main__':
    t0 = time()
    main()
    print('Execution time: {} s'.format(round((time() - t0), 2)))

    sys.exit(0)
