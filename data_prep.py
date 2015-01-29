#!/usr/bin/env python3
from __future__ import print_function
import glob
import os
from skimage.io import imread, imsave
from time import time
import numpy as np
import sys

CROP_SIZE = 64

TRAIN_DIR_IN = '/plankton/train'
TRAIN_DIR_OUT = '/plankton/train_fixed'


def resize_image(img):
    """Crops image up to CROP_SIZExCROP_SIZE and fills with white space
    if image is smaller than CROP_SIZExCROP_SIZE"""
    h, w = img.shape
    cropped = np.zeros((CROP_SIZE, CROP_SIZE), dtype=img.dtype) + 255
    start_y = (CROP_SIZE - h) // 2
    start_x = (CROP_SIZE - w) // 2
    new_h, new_w = min(h, CROP_SIZE), min(w, CROP_SIZE)
    cropped[max(start_y, 0):(max(start_y, 0) + new_h),
            max(start_x, 0):(max(start_x, 0) + new_w)] = img[max(-start_y, 0):(max(-start_y, 0) + new_h),
                                                             max(-start_x, 0):(max(-start_x, 0) + new_w)]
    return cropped


def main():
    script_start_time = time()
    if not os.path.isdir(TRAIN_DIR_IN):
        print('Error! No train dir found ("%s")' % TRAIN_DIR_IN, file=sys.stderr)
        return

    if os.path.isdir(TRAIN_DIR_OUT):
        print('Error! Train out file already exists ("%s")' % TRAIN_DIR_OUT, file=sys.stderr)
        return
    os.mkdir(TRAIN_DIR_OUT)

    classes = sorted(os.listdir(TRAIN_DIR_IN))
    for c in classes:
        print('Processing %s...' % c, end='')
        os.mkdir(os.path.join(TRAIN_DIR_OUT, c))
        files = glob.glob(os.path.join(TRAIN_DIR_IN, c, '*.jpg'))
        for f in files:
            img = imread(f, as_grey=True)
            cropped = resize_image(img)
            new_filename = os.path.join(TRAIN_DIR_OUT, c, os.path.basename(f))
            imsave(new_filename, cropped)
        print(' Ok!')

    print()
    print('Executed in %.1f seconds' % (time() - script_start_time))


if __name__ == '__main__':
    main()