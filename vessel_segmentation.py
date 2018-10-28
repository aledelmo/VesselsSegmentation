#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os.path
import sys
from time import time

import processing_vs as vs

__author__ = 'Alessandro Delmonte'
__email__ = 'delmonte.ale92@gmail.com'


def main():
    """
    Call to parser and pipeline

    :rtype: void
    """
    volume, arteries, veins, output = setup()
    if arteries is not None or veins is not None:
        vs.proc(volume, arteries, veins, output)


def setup():
    """
    Command line parser

    :rtype: list of strings
    :return: arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('Input_Volume', help='Name of the input reference volume', type=check_nii)
    parser.add_argument('-a', '--arteries', help='Pickled list of arteries initialization points', type=check_pickle)
    parser.add_argument('-v', '--veins', help='Pickled list of veins initialization points', type=check_pickle)
    parser.add_argument('Output_Volume', help='Name of the output reference volume', type=check_out)

    args = parser.parse_args()

    return args.Input_Volume, args.arteries, args.veins, args.Output_Volume


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


def check_out(value):
    """
       Verify that the file has a valid NIfTI-1 extension

       :param value: command line inserted string
       :return: file absolute path
       :rtype: string
       """
    if value.endswith('.nii') or value.endswith('.nii.gz'):
        return os.path.abspath(value)
    else:
        raise argparse.ArgumentTypeError("Invalid output extension (file format supported: nii, nii.gz): %s" % value)


def check_pickle(value):
    if value.endswith('.pkl') and os.path.isfile(os.path.abspath(value)):
        return os.path.abspath(value)
    else:
        raise argparse.ArgumentTypeError("Invalid initialization file (a pickled list is expected): %s" % value)


if __name__ == '__main__':
    t0 = time()
    main()
    print('Execution time: {} s'.format(round((time() - t0), 2)))

    sys.exit(0)
