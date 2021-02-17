from __future__ import division
import numpy as np
from scipy import ndimage
from renom.utility.image.data_augmentation.image import Image


def shift(x, shift, fill_mode="constant", fill_val=0, random=False, labels=None, num_class=0):
    """Performs shift transformation to Numpy images.
    if x is a Batch, apply shifts transform to Batch.
    if arguments include labels, apply label transformation.

    Args:
        x (ndarray): 3 or 4(batch) dimensional images
        shift (tuple): values of x and y shifts (y, x)
        fill_mode (str): method of interpolation after rotate transform
        fill_val (float): the interpolation value if fill_mode is 'constant'
        random (bool): random shift or not
        labels (ndarray): rectangle labels(2-dimensional array)
            ex:) np.array([[center x, center y, x_top_left, height, 0, 0, 0, 1, 0]])
        num_class (int): number of class of datasets

    Returns:
        (ndarray): Images(4 dimension) of shift transformed. If including labels, return with transformed labels

    Example:
        >>> from renom.utility.image.data_augmentation.shift import shift
        >>> from PIL import Image as im
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> image = im.open(image path) # ex:) "./image_folder/camera.jpg"
        >>> image = np.array(image).astype(np.float32)
        >>> shift_image = shift(image, shift=(50, 50))
        >>> fig, axes = plt.subplots(2, 1)
        >>> axes[0].imshow(image/255); axes[0].set_title("Original Image")
        >>> axes[1].imshow(shift_image[0] / 255); axes[1].set_title("Shift One Image")
        >>> plt.show()
    """
    shifts = Shift(shift, fill_mode=fill_mode, fill_val=fill_val)
    if isinstance(labels, np.ndarray):
        return shifts.transform(x, random=random, labels=labels, num_class=num_class)
    return shifts.transform(x, random=random)


class Shift(Image):
    """Apply shift transformation to the input x and labels

    Args:
        shift (tuple): values of x and y shifts (y, x)
        fill_mode (str): method of interpolation after rotate transform
        fill_val (float): the interpolation value if fill_mode is 'constant'
    """

    def __init__(self, shift, fill_mode="constant", fill_val=0):
        super(Shift, self).__init__()
        self.shift = shift
        self.fill_mode = fill_mode
        self.fill_val = fill_val

    def transform(self, x, random=0, labels=None, num_class=0):
        """Performs shift transformation to Numpy images.
        if x is a Batch, apply shifts transform to Batch.
        if arguments include labels, apply label transformation.

        Args:
            x (ndarray): 3 or 4(batch) dimensional images
            random (bool): random shift or not
            labels (ndarray): rectangle labels(2-dimensional array)
                               ex:) np.array([[center x, center y, x_top_left, height, 0, 0, 0, 1, 0]])
            num_class (int): number of class of datasets

        Returns:
            (ndarray): Images(4 dimension) of shift transformed. If including labels, return with transformed labels

        Example:
            >>> from renom.utility.image.data_augmentation.shift import Shift
            >>> from PIL import Image as im
            >>> import matplotlib.pyplot as plt
            >>> import numpy as np
            >>> image = im.open(image path) # ex:) "./image_folder/camera.jpg"
            >>> image = np.array(image).astype(np.float32)
            >>> sh = Shift(shift=(50, 50))
            >>> shift_image = sh.transform(image)
            >>> fig, axes = plt.subplots(2, 1)
            >>> axes[0].imshow(image/255); axes[0].set_title("Original Image")
            >>> axes[1].imshow(shift_image[0] / 255); axes[1].set_title("Shift One Image")
            >>> plt.show()

        """
        if self.shift == (0, 0):
            return x

        shift_images = self.check_x_dim(x.copy())[0]
        channel_axis = 3
        transformed_labels = None

        if random:
            shift_values = np.random.uniform(-1.0, 1.0, shift_images.shape[0])
            shift_images[:] = [self._get_shifted_image(
                image, channel_axis - 1, random=val) for val, image in zip(shift_values, shift_images)]
            if isinstance(labels, np.ndarray):
                transformed_labels = self._labels_transform(
                    labels, num_class, shift_values, x.shape[1:3])
        else:
            shift_images[:] = [self._get_shifted_image(
                image, channel_axis - 1) for image in shift_images]
            if isinstance(labels, np.ndarray):
                transformed_labels = self._labels_transform(
                    labels, num_class, np.ones((x.shape[0])), x.shape[1:3])

        if isinstance(labels, np.ndarray):
            return shift_images, transformed_labels
        return shift_images

    def _get_shifted_image(self, x, channel_axis, random=1.):
        transform_matrix = np.array([
            [1, 0, int(-self.shift[0] * random)],
            [0, 1, int(-self.shift[1] * random)],
            [0, 0, 1]
        ])
        x = np.rollaxis(x, channel_axis, 0)
        affine_matrix = transform_matrix[:2, :2]
        offset = transform_matrix[:2, 2]
        channel_images = [ndimage.interpolation.affine_transform(x_channel,
                                                                 affine_matrix,
                                                                 offset,
                                                                 order=0,
                                                                 mode=self.fill_mode,
                                                                 cval=self.fill_val) for x_channel in x]
        x = np.array(channel_images)
        x = np.rollaxis(x, 0, channel_axis + 1)
        return x

    def _labels_transform(self, labels, num_class, shift_values, img_shape):
        transform_labels = labels.copy()
        block_len = 4 + num_class
        for index, random, label in zip(np.arange(shift_values.shape[0]), shift_values, transform_labels):
            label = self._label_transform(label, block_len, random, img_shape)
            transform_labels[index] = label
        return transform_labels

    def _label_transform(self, label, block_len, random, img_shape):
        num_block = label.shape[0] // block_len
        x_shift = int(self.shift[1] * random)
        y_shift = int(self.shift[0] * random)
        label_num = 0
        for block in range(num_block):
            if label[block * (block_len):(block + 1) * (block_len)][2:4].all() == 0.:
                label[label_num * block_len:] = 0.
                return label
            center_x, center_y, label_width, label_height = label[block * (
                block_len):(block + 1) * (block_len)][:4]
            center_x += x_shift
            center_y += y_shift
            if x_shift > 0:
                if (center_x - label_width / 2.) >= img_shape[1]:
                    continue
                if (center_x + label_width / 2.) >= img_shape[1]:
                    diff_width = center_x + label_width / 2. - img_shape[1] + 1
                    label_width -= diff_width
                    center_x -= diff_width / 2.
            elif x_shift < 0:
                if (center_x + label_width / 2.) <= 0:
                    continue
                if (center_x - label_width / 2.) < 0:
                    diff_width = label_width / 2. - center_x
                    label_width -= diff_width
                    center_x += diff_width / 2.

            if y_shift > 0:
                if (center_y - label_height / 2.) >= img_shape[0]:
                    continue
                if (center_y + label_height / 2.) > img_shape[0]:
                    diff_height = center_y + \
                        label_height / 2 - img_shape[0] + 1
                    label_height -= diff_height
                    center_y -= diff_height / 2.
            elif y_shift < 0:
                if (center_y + label_height / 2.) <= 0:
                    continue
                if (center_y - label_height / 2.) < 0:
                    diff_height = label_height / 2. - center_y
                    label_height -= diff_height
                    center_y += diff_height / 2.

            if (label_width > 0) and (label_height > 0):
                label[label_num * (block_len):(label_num + 1) * (block_len)
                      ][:4] = center_x, center_y, label_width, label_height
                label[label_num * (block_len):(label_num + 1) * (block_len)
                      ][4:] = label[block * (block_len):(block + 1) * (block_len)][4:]
                label_num += 1
        label[label_num * block_len:] = 0.
        return label
