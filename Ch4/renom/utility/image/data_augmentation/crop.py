from __future__ import print_function
from __future__ import division
import numpy as np
from renom.utility.image.data_augmentation.image import Image


def crop(x, left_top=(0, 0), size=(0, 0), labels=None, num_class=0, random=False):
    """Performs a Crop of a Numpy images.
    if x is a Batch, apply Crop transform to Batch.
    if arguments include labels, apply label transformation.

    Args:
        x (ndarray): 3 or 4(batch) dimensional images
        left_top (tuple): x and y of top left (y, x)
        size (tuple): width and height of crop image (Height, Width)
        labels (ndarray): rectangle labels(2-dimensional array)
                           ex:) np.array([[center x, center y, x_top_left, height, 0, 0, 0, 1, 0]])
        num_class (int): number of class of datasets (for rectangle transformation)
        random (bool): If True, apply random cropping. left_top is randomly decided.

    Returns:
        (ndarray): Images(4 dimension) of crop transformed. If including labels, return with transformed labels

    Example:
        >>> from renom.utility.image.data_augmentation.crop import crop
        >>> from PIL import Image as im
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> image = im.open(image path) # ex:) "./image_folder/camera.jpg"
        >>> image = np.array(image).astype(np.float32)
        >>> crop_image = crop(image, left_top=(10, 10), size=(100, 100))
        >>> fig, axes = plt.subplots(2, 1)
        >>> axes[0].imshow(image/255); axes[0].set_title("Original Image")
        >>> axes[1].imshow(crop_image[0] / 255); axes[1].set_title("Crop One Image")
        >>> plt.show()

    """
    crop = Crop(left_top=left_top, size=size)
    if isinstance(labels, np.ndarray):
        return crop.transform(x, random=random, labels=labels, num_class=num_class)
    return crop.transform(x, random=random)


class Crop(Image):
    """Apply crop transformation to the input x and labels.

    Args:
        left_top (tuple): x and y of top left (y, x)
        size (tuple): width and height of crop image (Height, Width)
    """

    def __init__(self, left_top=(0, 0), size=(0, 0)):
        self.left_top = left_top
        self.size = size

    def transform(self, x, random=False, labels=None, num_class=0):
        """Performs a Crop of a Numpy images.
        if x is a Batch, apply Crop transform to Batch
        if arguments include labels, apply label transformation

        Args:
            x (ndarray): 3 or 4(batch) dimensional images
            random (bool): If True, apply random cropping. left_top is randomly desided.
            labels (ndarray): rectangle labels(2-dimensional array)
                               ex:) np.array([[center x, center y, x_top_left, height, 0, 0, 0, 1, 0]])
            num_class (int): number of class of datasets (for rectangle transformation)

        Returns:
            (ndarray): Images(4 dimension) of crop transformed. If including labels, return with transformed labels

        Example:
            >>> from renom.utility.image.data_augmentation.crop import Crop
            >>> from PIL import Image as im
            >>> import matplotlib.pyplot as plt
            >>> import numpy as np
            >>> image = im.open(image path) # ex:) "./image_folder/camera.jpg"
            >>> image = np.array(image).astype(np.float32)
            >>> cr = Crop(left_top=(10, 10), size=(100, 100))
            >>> crop_image = cr.transform(image)
            >>> fig, axes = plt.subplots(2, 1)
            >>> axes[0].imshow(image/255); axes[0].set_title("Original Image")
            >>> axes[1].imshow(crop_image[0] / 255); axes[1].set_title("Crop One Image")
            >>> plt.show()

        """

        batch_x, batch_size, original_size = self.check_x_dim(x)

        try:
            self._check_cropsize(batch_x)
        except ValueError as e:
            print("In crop.py ", e)
            print("return Original x")
            if isinstance(labels, np.ndarray):
                return x, labels
            else:
                return x

        if random:
            cropped_images = np.zeros(
                (batch_x.shape[0], self.size[0], self.size[1], batch_x.shape[3]), dtype=batch_x.dtype)
            y_top_lefts = batch_x.shape[1] - self.size[0]
            x_top_lefts = batch_x.shape[2] - self.size[1]
            y_top_lefts = np.random.randint(0, y_top_lefts, batch_x.shape[0])
            x_top_lefts = np.random.randint(0, x_top_lefts, batch_x.shape[0])
            cropped_images[:] = [image[y_top_left:y_top_left + self.size[0], x_top_left:x_top_left + self.size[1]]
                                 for y_top_left, x_top_left, image in zip(y_top_lefts, x_top_lefts, batch_x)]
            if isinstance(labels, np.ndarray):
                new_labels = self._labels_transform(
                    labels, y_top_lefts, x_top_lefts, num_class)
                return cropped_images, new_labels
            return cropped_images

        if isinstance(labels, np.ndarray):
            new_labels = self._labels_transform(labels, np.ones(
                (batch_x.shape[0])) * self.left_top[0], np.ones((batch_x.shape[0])) * self.left_top[1], num_class)
            return batch_x[:, self.left_top[0]:self.left_top[0] + self.size[0],
                           self.left_top[1]:self.left_top[1] + self.size[1]].copy(), new_labels
        else:
            return batch_x[:, self.left_top[0]:self.left_top[0] + self.size[0],
                           self.left_top[1]:self.left_top[1] + self.size[1]].copy()

    def _labels_transform(self, labels, y_top_lefts, x_top_lefts, num_class):
        """Perform labels transformation for rectangle

            (center x, center y, x_top_left, height, 0, 0, 0, 1, 0) * num of rectangle object
        """
        transform_labels = labels.copy()
        block_len = 4 + num_class
        if isinstance(y_top_lefts, np.ndarray):
            for index, label, y_top_left, x_top_left in zip(np.arange(labels.shape[0]),
                                                            transform_labels, y_top_lefts, x_top_lefts):
                label = self._label_transform(
                    label, y_top_left, x_top_left, block_len)
                transform_labels[index] = label
        else:
            transform_labels[0] = self._label_transform(
                transform_labels[0], y_top_lefts, x_top_lefts, block_len)
        return transform_labels

    def _label_transform(self, label, y_top_left, x_top_left, block_len):
        num_block = len(label) // block_len
        label_num = 0
        for block in range(num_block):
            if label[block * (block_len):(block + 1) * (block_len)][2:4].all() == 0.:
                label[label_num * block_len:] = 0.
                return label

            center_x, center_y, label_width, label_height = label[block * (
                block_len):(block + 1) * (block_len)][:4]
            label_x = center_x - label_width / 2.  # label's width of top left
            label_y = center_y - label_height / 2.  # label's height of top left
            if label_x < x_top_left:
                if label_x + label_width <= x_top_left:
                    continue
                label_width -= (x_top_left - label_x)
                label_x = 0
            elif x_top_left + self.size[1] <= label_x:
                continue
            else:
                label_x -= x_top_left

            if (label_x + label_width) >= self.size[1]:
                label_width -= ((label_x + label_width) - self.size[1] + 1)

            if label_y < y_top_left:
                if label_y + label_height <= y_top_left:
                    continue
                label_height -= ((y_top_left - label_y))
                label_y = 0
            elif y_top_left + self.size[0] <= label_y:
                continue
            else:
                label_y -= y_top_left

            if (label_y + label_height) >= self.size[0]:
                label_height -= ((label_y + label_height) - self.size[0] + 1)

            if (label_width > 0) and (label_height > 0):
                center_x = label_x + label_width / 2.
                center_y = label_y + label_height / 2.
                label[label_num * (block_len):(label_num + 1) * (block_len)
                      ][:4] = center_x, center_y, label_width, label_height
                label[label_num * (block_len):(label_num + 1) * (block_len)
                      ][4:] = label[block * (block_len):(block + 1) * (block_len)][4:]
                label_num += 1
        label[label_num * block_len:] = 0.
        return label

    def _check_cropsize(self, x):
        shape = x.shape[1:]

        if 0 in self.size:
            raise ValueError("Please set size over 0")
        if ((shape[0] < (self.left_top[0] + self.size[0])) or (shape[1] < (self.left_top[1] + self.size[1]))):
            raise ValueError("Please set place and size within image shape")
