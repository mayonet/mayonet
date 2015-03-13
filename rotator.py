from itertools import product
from random import choice, randint
import numpy as np
from pylearn2.datasets.preprocessing import CentralWindow
from pylearn2.train_extensions import TrainExtension
from skimage.transform import rotate, rescale
import data_prep
from skimage.filter import rank
from skimage.morphology import disk
import multiprocessing

from PIL import Image


class RotatorExtension(TrainExtension):

    def __init__(self, window, randomize, center=(),
                 angles=range(360),
                 x_offsets=(0, 1), y_offsets=(0, 1),
                 median_radii=(0,),
                 mean_radii=(0,),
                 flip=True,
                 scales=(1,)):
        self.window = window
        self.randomize = randomize
        self.center = center
        self.angles = angles
        self.median_radii = median_radii
        self.mean_radii = mean_radii
        self.flip = flip
        self.scales = scales
        self.originals = [ds.get_topological_view() for ds in randomize]
        self.offsets = list(product(x_offsets, y_offsets))
        if len(randomize) > 0:
            self.originals = [ds.get_topological_view() for ds in randomize]
            self.init_image_shape = randomize[0].get_topological_view()[0].shape[:2]
        else:
            self.originals = []
            self.init_image_shape = center[0].get_topological_view()[0].shape[:2]
        self.max_offset = [d-w for d, w in zip(self.init_image_shape, self.window)]
        self.center_offset = [d//2 for d in self.max_offset]

    def change_images(self, initial_images, fn):
        return np.array([fn(img) for img in initial_images], dtype='float32')

    def setup(self, model, dataset, algorithm):
        for data in self.center:
            data.set_topological_view(self.change_images(data.get_topological_view(), lambda img: self.crop(img, self.center_offset, self.window)))

        self.randomize_datasets()

    def randomize_datasets(self):
        for original, dataset in zip(self.originals, self.randomize):
            imgs = original
            new_imgs = np.array([self.randomize_image(img) for img in imgs], dtype='float32')

            dataset.set_topological_view(new_imgs, axes=dataset.view_converter.axes)

    def on_monitor(self, model, dataset, algorithm):
        model = None
        dataset = None
        algorithm = None

        self.randomize_datasets()

    def randomize_image(self, img):
        h, w = img.shape[:2]
        img = rescale(img, choice(self.scales))
        img = rotate(img, choice(self.angles))
        if self.flip and randint(0, 1) == 0:
            img = np.fliplr(img)
        img = (255*(1-img[:, :, 0])).astype('uint8')

        med = choice(self.median_radii)
        if med > 0:
            img = rank.median(img, disk(med))
        mea = choice(self.mean_radii)
        if mea > 0:
            img = rank.mean(img, disk(mea))
        # workaround
        img = data_prep.crop(img, [h, w])
        img = 1-(img[:, :, np.newaxis]/255.)
        if self.init_image_shape != self.window:
            offset = [c1//2 + r for c1, r in zip(self.max_offset, choice(self.offsets))]
            img = self.crop(img, offset, self.window)
        return img

    def crop(self, img, offset, new_size):
        new_x, new_y = new_size
        off_x, off_y = offset
        return img[off_x:(off_x + new_x), off_y:(off_y + new_y), :]


def crop_01(img, offset, new_size):
    new_x, new_y = new_size
    off_x, off_y = offset
    return img[off_x:(off_x + new_x), off_y:(off_y + new_y)]


def basic_crop(img, new_sizes):
    h, w = img.shape
    cropped = np.zeros(new_sizes, dtype=img.dtype)
    start_y = (new_sizes[0] - h) // 2
    start_x = (new_sizes[1] - w) // 2
    new_h, new_w = min(h, new_sizes[0]), min(w, new_sizes[1])
    cropped[max(start_y, 0):(max(start_y, 0) + new_h),
            max(start_x, 0):(max(start_x, 0) + new_w)] = img[max(-start_y, 0):(max(-start_y, 0) + new_h),
                                                             max(-start_x, 0):(max(-start_x, 0) + new_w)]
    return cropped


def randomize_image_c01(img, window, angles=(0,), x_offsets=(0,), y_offsets=(0,),
                        median_radii=(0,), mean_radii=(0,), flip=False, scales=(1,)):
    offsets = list(product(y_offsets, x_offsets))
    img = img[0]
    h, w = img.shape
    # im = Image.fromarray(np.cast['uint8'](img*255))
    # scale = choice(scales)
    # im = im.resize((int(w*scale), int(h*scale)))
    # im = im.rotate(choice(angles))
    # img = np.asarray(im, dtype='float32')/255.
    img = rescale(img, choice(scales))
    img = rotate(img, choice(angles))
    if flip and randint(0, 1) == 0:
        img = np.fliplr(img)

    img = basic_crop(img, [h, w])

    max_offset = [d-w for d, w in zip(img.shape, window)]
    if img.shape != window:
        offset = [c1//2 + r*choice([1, -1]) for c1, r in zip(max_offset, choice(offsets))]
        img = crop_01(img, offset, window)

    med = choice(median_radii)
    if med > 0:
        img = rank.median(img, disk(med))
    mea = choice(mean_radii)
    if mea > 0:
        img = rank.mean(img, disk(mea))
    return img[np.newaxis, :, :]

# t0 = time()
# batch_x = randomize_dataset(train_x, (96, 96), scales=(2, 0.5))
# print(time() - t0)

def randomize_dataset_bc01(imgs, window, angles=range(360), x_offsets=(0, 1), y_offsets=(0, 1),
                           median_radii=(0,), mean_radii=(0,), flip=True, scales=(1,)):
    return np.array([randomize_image_c01(img, window, angles,
                                         x_offsets, y_offsets,
                                         median_radii, mean_radii,
                                         flip, scales) for img in imgs], dtype='float32')