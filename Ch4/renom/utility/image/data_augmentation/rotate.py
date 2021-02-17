from __future__ import division
import numpy as np
from scipy import ndimage
from renom.utility.image.data_augmentation.image import Image


def rotate(x, degree, fill_mode="constant", fill_val=0, random=False, labels=None):
    """Performs a rotation of a Numpy images.
    if x is a Batch, apply rotation transform to Batch.

    Args:
        x (ndarray): 3 or 4(batch) dimensional images
        degree (int): rotation degree [-180 : 180]
        fill_mode (str): method of interpolation after rotate transform
            you can use ['constant', 'nearest']
        fill_val (float): the interpolation value if fill_mode is 'constant'
        random (bool): random rotation. degree is [-degree, +degree]
        labels (ndarray): you can use only when degree=90

    Returns:
        (ndarray): Images(4 dimension) of rotate transformed.
            If including labels, return with transformed labels

    Example:
        >>> from renom.utility.image.data_augmentation.rotate import rotate
        >>> from PIL import Image as im
        >>> import numpy as np
        >>> image = im.open(image path) # ex:) "./image_folder/camera.jpg"
        >>> image = np.array(image).astype(np.float32)
        >>> rotate_image = rotate(image, degree=10)
    """
    rotate = Rotate(degree, fill_mode=fill_mode, fill_val=fill_val)
    return rotate.transform(x, random=random, labels=labels)


class Rotate(Image):
    """Apply rotate transformation to the input x

    Args:
        x (ndarray): 3 or 4(batch) dimensional images
        degree (int): rotation degree [-180 : 180]
        fill_mode (str): method of interpolation after rotate transform
                          you can use ['constant', 'nearest']
        fill_val (float): the interpolation value if fill_mode is 'constant'

    Example:
        >>> from rotate import rotate
        >>> from PIL import Image as im
        >>> import numpy as np
        >>> image = im.open(image path) # ex:) "./image_folder/camera.jpg"
        >>> image = np.array(image).astype(np.float32)
        >>> rt = Rotate(degree=10)
        >>> rotate_image = rt.transform(image)
    """

    def __init__(self, degree, fill_mode="constant", fill_val=0):
        super(Rotate, self).__init__()
        self.degree = degree
        self.fill_mode = fill_mode
        self.fill_val = fill_val

    def transform(self, x, random=False, labels=None, num_class=0, **kwargs):
        """Performs a rotation of a Numpy images.
        if x is a Batch, apply rotation transform to Batch

        Args:
            x (ndarray): 3 or 4(batch) dimensional images
            random (bool): random rotation. degree is [-degree, +degree]
            labels (ndarray): you can use only when degree=90

        Retuens:
            (ndarray): Images(4 dimension) of rotate transformed.
              If including labels, return with transformed labels
        """

        rotate_images, batch_size, original_size = self.check_x_dim(x.copy())
        row_axis, col_axis, channel_axis = 1, 2, 3

        center = rotate_images.shape[row_axis] // 2, \
            rotate_images.shape[col_axis] // 2

        if isinstance(labels, np.ndarray):
            """
            If given np.array of labels (detection bounding boxes),
            rotation only supports 90deg rotations.
            """
            if self.degree != 90:
                raise NotImplementedError(
                    "Label transformation only support 90deg rotation angle.")
            if random:
                shuffle = np.random.randint(0, 2, batch_size)
                rotate_images[:] = [self._get_rotated_image(
                    image, center, channel_axis - 1, random=val) for val, image in zip(shuffle, rotate_images)]

                transformed_labels = self._labels_transform(
                    labels, num_class, np.where(shuffle)[0], original_size)
            else:
                rotate_images[:] = [self._get_rotated_image(
                    image, center, channel_axis - 1) for image in rotate_images]
                transformed_labels = self._labels_transform(
                    labels, num_class, np.arange(batch_size), original_size)

            return rotate_images, transformed_labels

        else:
            """
            If given no labels, rotation supports any angle betweeen -180 and 180deg.
            """
            if random:
                shuffle = np.random.uniform(-1.0, 1.0, rotate_images.shape[0])
                rotate_images[:] = [self._get_rotated_image(
                    image, center, channel_axis - 1, random=val) for val, image in zip(shuffle, rotate_images)]
            else:
                rotate_images[:] = [self._get_rotated_image(
                    image, center, channel_axis - 1) for image in rotate_images]

            return rotate_images

    def _cal_transform_matrix(self, center, random=1):
        angle = random * np.pi * self.degree / 180.0
        rotate_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

        warp_matrix = np.array([
            [1, 0, center[0]],
            [0, 1, center[1]],
            [0, 0, 1]
        ])
        unwarp_matrix = np.array([
            [1, 0, -center[0]],
            [0, 1, -center[1]],
            [0, 0, 1]
        ])
        return np.dot(np.dot(warp_matrix, rotate_matrix), unwarp_matrix)

    def _get_rotated_image(self, image, center, channel_axis, random=1):
        transform_matrix = self._cal_transform_matrix(center, random=random)
        image = np.rollaxis(image, channel_axis, 0)
        affine_matrix = transform_matrix[:2, :2]
        offset = transform_matrix[:2, 2]
        channel_images = [ndimage.interpolation.affine_transform(x_channel, affine_matrix,
                                                                 offset,
                                                                 order=0,
                                                                 mode=self.fill_mode,
                                                                 cval=self.fill_val) for x_channel in image]
        image = np.array(channel_images)
        image = np.rollaxis(image, 0, channel_axis + 1)
        return image

    def _labels_transform(self, labels, num_class, shuffle, img_shape):
        transformed_labels = labels.copy()
        block_len = 4 + num_class
        num_block = labels[0].shape[0] // block_len
        for index, label in zip(shuffle, transformed_labels[shuffle]):
            for block in range(num_block):
                if label[block * (block_len):(block + 1) * (block_len)][2:4].all() == 0.:
                    break
                x, y, w, h = label[block *
                                   (block_len):(block + 1) * (block_len)][0:4]
                new_h = w
                new_w = h
                new_x = img_shape[1] - y
                new_y = x
                label[block * (block_len):(block + 1) * (block_len)
                      ][0:4] = [new_x, new_y, new_w, new_h]
            transformed_labels[index] = label
        return transformed_labels
