#!/usr/bin/env python3
from __future__ import print_function
import glob
import os
from skimage.io import imread, imsave
from skimage.transform import resize
from time import time
import numpy as np
import sys
# from wand.image import Image

CROP_SIZE = 80

TRAIN_DIR_IN = '/plankton/train'
TRAIN_DIR_OUT = '/plankton/train_fixed'


def crop(img, new_sizes=(CROP_SIZE, CROP_SIZE)):
    """Crops image up to CROP_SIZExCROP_SIZE and fills with white space
    if image is smaller than CROP_SIZExCROP_SIZE"""
    h, w = img.shape
    cropped = np.zeros(new_sizes, dtype=img.dtype) + 255
    start_y = (new_sizes[0] - h) // 2
    start_x = (new_sizes[1] - w) // 2
    new_h, new_w = min(h, new_sizes[0]), min(w, new_sizes[1])
    cropped[max(start_y, 0):(max(start_y, 0) + new_h),
            max(start_x, 0):(max(start_x, 0) + new_w)] = img[max(-start_y, 0):(max(-start_y, 0) + new_h),
                                                             max(-start_x, 0):(max(-start_x, 0) + new_w)]
    return cropped


def blunt_resize(img, new_size):
    return resize(img, (new_size, new_size), cval=255)*255


def crop_white(img):
    h, w = img.shape
    xs = np.min(img, 0)
    ys = np.min(img, 1)

    left = 0
    right = w-1
    top = 0
    bottom = h-1

    for i in range(w):
        if xs[i] < 255:
            left = i
            break

    for i in range(h):
        if ys[i] < 255:
            top = i
            break

    for i in range(w, 0, -1):
        if xs[i-1] < 255:
            right = i
            break

    for i in range(h, 0, -1):
        if ys[i-1] < 255:
            bottom = i
            break

    return img[top:bottom, left:right]


def blunt_after_crop(img, new_size):
    return blunt_resize(crop_white(img), new_size)


def shrink(img, new_size):
    h, w = img.shape
    res = crop(img, (max(h, new_size), max(w, new_size)))
    return blunt_resize(res, new_size)


def rational_resize(img, new_size):
    h, w = img.shape
    t = max(h, w)
    return blunt_resize(crop(img, (t, t)), new_size)


def resize_image(img, size=CROP_SIZE, method='crop'):
    if method == 'crop':
        return crop(img, (size, size))
    elif method == 'bluntresize':
        return blunt_resize(img, size)
    elif method == 'shrink':
        return shrink(img, size)
    elif method == 'rationalresize':
        return rational_resize(img, size)


def read_image(fn, size, method):
    # if method == 'liquid':
    #     new_img = np.zeros((size, size))
    #     with Image(filename=fn) as img:
    #         img.liquid_rescale(size, size)
    #         for y, row in enumerate(img):
    #             for x, elem in enumerate(row):
    #                 new_img[y, x] = elem.red_int8
    #
    # else:
    img = imread(fn, as_grey=True)

    if method == 'crop':
        new_img = crop(img, (size, size))
    elif method == 'bluntresize':
        new_img = blunt_resize(img, size)
    elif method == 'shrink':
        new_img = shrink(img, size)
    elif method == 'rationalresize':
        new_img = rational_resize(img, size)
    elif method == 'bluntaftercrop':
        new_img = blunt_after_crop(img, size)
    else:
        raise Exception('Illegal method "%s"' % method)

    return new_img.reshape((size * size,))

    # return resize_image(imread(fn, as_grey=True), size, method).reshape((size * size,))


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