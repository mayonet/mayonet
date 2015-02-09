from itertools import product
from random import choice, randint
import numpy as np
from pylearn2.datasets.preprocessing import CentralWindow
from pylearn2.train_extensions import TrainExtension
from skimage.transform import rotate, rescale
import data_prep


class Rotator(TrainExtension):

    def __init__(self, window, randomize, center=(), angles=range(360), x_offsets=(0, 1), y_offsets=(0, 1), flip=True,
                 scales=(1,)):
        self.window = window
        self.randomize = randomize
        self.center = center
        self.angles = angles
        self.flip = flip
        self.scales = scales
        self.originals = [ds.get_topological_view() for ds in randomize]
        self.offsets = list(product(x_offsets, y_offsets))
        if len(randomize) > 0:
            self.init_image_shape = randomize[0].get_topological_view()[0].shape[:2]
        else:
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
        # Костыли-костылики
        img = data_prep.crop(img[:, :, 0]+255, [h, w])[:, :, np.newaxis]-255
        if self.init_image_shape != self.window:
            offset = [c1//2 + r for c1, r in zip(self.max_offset, choice(self.offsets))]
            img = self.crop(img, offset, self.window)
        return img

    def crop(self, img, offset, new_size):
        new_x, new_y = new_size
        off_x, off_y = offset
        return img[off_x:(off_x + new_x), off_y:(off_y + new_y), :]