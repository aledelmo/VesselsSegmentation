#!/usr/bin/env python
# -*- coding: utf-8 -*-

from time import time
import argparse
import sys
import os.path
import processing_vs as vs

__author__ = 'Alessandro Delmonte'
__email__ = 'delmonte.ale92@gmail.com'


def main():
    """
    Call to parser and pipeline
    """
    volume, seeds, vessel_type, output = setup()

    vs.proc(volume, seeds, vessel_type, output)


def setup():
    """
    Command line parser
    :return: arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('Input_Volume', help='Name of the input reference volume', type=check_nii)
    parser.add_argument('Initialization', help='Pickled list of initialization points', type=check_pickle)
    parser.add_argument('Type', help='Choose between Artery and Vein', choices=['Artery', 'artery', 'Vein', 'vein'])
    parser.add_argument('Output_Volume', help='Name of the input reference volume', type=check_out)

    args = parser.parse_args()

    return args.Input_Volume, args.Initialization, args.Type, args.Output_Volume


def check_nii(value):
    if value.endswith('.nii') or value.endswith('.nii.gz') and os.path.isfile(os.path.abspath(value)):
        return os.path.abspath(value)
    else:
        raise argparse.ArgumentTypeError("Invalid output extension (file format supported: nii, nii.gz): %s" % value)


def check_out(value):
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
