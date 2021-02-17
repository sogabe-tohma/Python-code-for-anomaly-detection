# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np

from renom.utility.distributor.imageloader import ImageLoader
from renom.utility.image.data_augmentation.resize import resize
from .utilities import make_ndarray


class ImageDistributor(object):
    """Base class for image distribution.
    Use subclasses ImageClassificationDistributor,
    ImageDetectionDistributor, ImageSegmentationDistributor depending
    on the image task. Or sublass it for original image tasks.

    Args:
        image_path_list (list): List of image path.
        y_list (list): List of labels (bbox and class) for every image (2 dimensional array).
        class_list (list): List of classes name for this dataset.
        shuffle (bool): If True, apply datasets shuffle per epoch
        imsize (tuple): Resize input image for converting batch ndarray.
        color (str): Color of Input Image. ["RGB", "GRAY"]
        augmentation (function): Augmentater for input Image.
    """

    def __init__(self, image_path_list, y_list=None, class_list=None, imsize=(32, 32), color="RGB", augmentation=None):
        self._data_table = image_path_list
        self._data_size = len(image_path_list)
        self._data_y = y_list
        self._class_list = class_list
        self._imsize = imsize
        self._color = color
        self._augmentation = augmentation

    def __len__(self):
        return self._data_size


class ImageDetectionDistributor(ImageDistributor):
    """Distributor class for tasks of image detection.
    Labels are expected to be Bounding boxes and Classes.
    ex:) np.array([[center x, center y, x_top_left, height, 0, 0, 0, 1, 0]])

    Args:
        image_path_list (list): list of image path
        y_list (list): list of labels (bbox and class) for every image
        class_list (list): list of classes name for this dataset
        shuffle (bool): If True, apply datasets shuffle per epoch
        imsize (tuple): resize input image for converting batch ndarray
        color (str): color of Input Image. ["RGB", "GRAY"]
        augmentation (function): augmentater for Input Image

    :Example:
        >>> from renom.utility.load.imageloader.threadingdistributor import ImageDetectionDistributor
        >>> from renom.utility.image.data_augmentation import *
        >>> datagenerator = DataAugmentation([
        ...     Flip(1),
        ...     Rotate(20),
        ...     Crop(size=(300, 300)),
        ...     Resize(size=(500, 500)),
        ...     Shift((20, 50)),
        ...     Color_jitter(v = (0.5, 2.0)),
        ...     Zoom(zoom_rate=(1.2, 2))
        ...     # Rescale(option='zero'),
        ... ], random = True)
        >>> dist = ImageDetectionDistributor(x_list, y_list=y_list,
                                        class_list=class_list,callback=datagenerator,
                                        shuffle=True, imsize=(360, 360), color='RGB')
        >>> for i, (x, y) in enumerate(dist.batch(32)):
        ...     print 'Batch', i
    """

    def __init__(self, image_path_list, y_list=None, class_list=None, imsize=(360, 360),
                 color='RGB', augmentation=None):
        super(ImageDetectionDistributor, self).__init__(image_path_list, y_list=y_list,
                                                        class_list=class_list, imsize=imsize,
                                                        color=color, augmentation=augmentation)

        if self._data_y is not None:
            self._data_y, _ = make_ndarray(self._data_y, len(self._class_list))

    def batch(self, batch_size, shuffle):
        """Returns generator of batch images.

        Args:
            batch_size (int): size of a batch.
        Returns:
            (ndarray): Images(4 dimension) of input data for Network.
              If including labels, return with transformed labels
        """
        if shuffle:
            perm = np.random.permutation(self._data_size)
        else:
            perm = np.arange(self._data_size)
        batches = [perm[i * batch_size:(i + 1) * batch_size]
                   for i in range(int(np.ceil(self._data_size / batch_size)))]

        imgfiles = [[self._data_table[p] for p in b] for b in batches]
        imgs = ImageLoader(imgfiles, self._color)
        for p, imgs in zip(batches, imgs.wait_images()):
            # Case: we are given both images and labels
            if self._data_y is not None:
                data_y = self._data_y[p].copy()
                for index, img in enumerate(imgs):
                    label = np.array([data_y[index]], dtype=np.float32)
                    if self._color == "GRAY":
                        img = np.array(img, dtype=np.float32)[:, :, np.newaxis]
                    else:
                        img = np.array(img, dtype=np.float32)
                    image, data_y[index] = resize(
                        img, size=self._imsize, labels=label, num_class=len(self._class_list))
                    imgs[index] = image[0]
                if self._augmentation is not None:
                    imgs, data_y = self._augmentation.create(
                        np.array(imgs, dtype=np.float32), labels=data_y, num_class=len(self._class_list))
                imgs = np.array(imgs, dtype=np.float32).transpose((0, 3, 1, 2))
                yield imgs, data_y
            # Case: we are only given images
            else:
                for index, img in enumerate(imgs):
                    if self._color == "GRAY":
                        img = np.array(img, dtype=np.float32)[:, :, np.newaxis]
                    else:
                        img = np.array(img, dtype=np.float32)
                    imgs[index] = resize(img, size=self._imsize)[0]
                if self._augmentation is not None:
                    imgs = self._augmentation.create(np.array(imgs, dtype=np.float32))
                imgs = np.array(imgs, dtype=np.float32).transpose((0, 3, 1, 2))
                yield imgs


class ImageClassificationDistributor(ImageDistributor):
    """Distributor class for tasks of image classification.

    Args:
        image_path_list (list): list of image path
        y_list (list): list of labels (bbox and class) for every image
        class_list (list): list of classes name for this dataset
        shuffle (bool): If True, apply datasets shuffle per epoch
        imsize (tuple): resize input image for converting batch ndarray
        color (str): color of Input Image. ["RGB", "GRAY"]
        augmentation: (function) augmentater for Input Image

    Example:
        >>> from renom.utility.load.imageloader.threadingdistributor import ImageClassificationDistributor
        >>> from renom.utility.image.data_augmentation import *
        >>> datagenerator = DataAugmentation([
        ...     Flip(1),
        ...     Rotate(20),
        ...     Crop(size=(300, 300)),
        ...     Resize(size=(500, 500)),
        ...     Shift((20, 50)),
        ...     Color_jitter(v = (0.5, 2.0)),
        ...     Zoom(zoom_rate=(1.2, 2))
        ...     # Rescale(option='zero'),
        ... ], random = True)
        >>> dist = ImageClassificationDistributor(x_list, y_list=y_list,
                                                class_list=class_list, callback=datagenerator,
                                                shuffle=True, imsize=(360, 360), color='RGB')
        >>> for i, (x, y) in enumerate(dist.batch(32)):
        ...     print 'Batch', i
    """

    def __init__(self, image_path_list, y_list=None, class_list=None,
                 imsize=(360, 360), color='RGB', augmentation=None):
        super(ImageClassificationDistributor, self).__init__(image_path_list, y_list=y_list,
                                                             class_list=class_list, imsize=imsize,
                                                             color=color, augmentation=augmentation)

    def batch(self, batch_size, shuffle):
        """
        Args:
            batch_size (int): size of a batch.
        Returns:
            (ndarray): Images(4 dimension) of input data for Network. If including labels, return with original labels
        """
        if shuffle:
            perm = np.random.permutation(self._data_size)
        else:
            perm = np.arange(self._data_size)
        batches = [perm[i * batch_size:(i + 1) * batch_size]
                   for i in range(int(np.ceil(self._data_size / batch_size)))]

        imgfiles = [[self._data_table[p] for p in b] for b in batches]
        labels = [[self._data_y[p] for p in b] for b in batches]

        imgs = ImageLoader(imgfiles, self._color)

        for lbls, imgs in zip(labels, imgs.wait_images()):
            for index, img in enumerate(imgs):
                if self._color == "GRAY":
                    img = np.array(img, dtype=np.float32)[:, :, np.newaxis]
                else:
                    img = np.array(img, dtype=np.float32)
                imgs[index] = resize(img, size=self._imsize)[0]
            if self._augmentation is not None:
                imgs = self._augmentation.create(np.array(imgs, dtype=np.float32))
            lbls = np.array(lbls)
            imgs = np.array(imgs, dtype=np.float32).transpose(0, 3, 1, 2)
            if self._data_y is None:
                yield imgs
            yield imgs, lbls
