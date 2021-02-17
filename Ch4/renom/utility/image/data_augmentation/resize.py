from __future__ import print_function
from __future__ import division
import numpy as np
from renom.utility.image.data_augmentation.image import Image
try:
    import cv2
    with_cv2 = True
except Exception:
    from skimage.transform import resize as rs
    with_cv2 = False


def resize(x, size=(0, 0), labels=None, num_class=0, **kwargs):
    """Performs a resize transformation of a Numpy Image 'x'.
    if x is a Batch, apply Resize transform to Batch.
    if arguments include labels, apply label transformation.

    :param ndarray x: 3 or 4(batch) dimensional x(images)
    :param ndarray labels: rectangle labels(2-dimensional array)
                           ex:) np.array([[center x, center y, x_top_left, height, 0, 0, 0, 1, 0]])
    :param int num_class: number of class of datasets
    :return: Images(4 dimension) of resize transformed. If including labels, return with transformed labels
    :rtype: ndarray

    :Example:
    >>> from renom.utility.image.data_augmentation.resize import resize
    >>> from PIL import Image as im
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> image = im.open(image path) # ex:) "./image_folder/camera.jpg"
    >>> image = np.array(image).astype(np.float32)
    >>> resize_image = resize(image, size=(300, 500))
    >>> fig, axes = plt.subplots(2, 1)
    >>> axes[0].imshow(image/255); axes[0].set_title("Original Image")
    >>> axes[1].imshow(resize_image[0] / 255); axes[1].set_title("Resize One Image")
    >>> plt.show()

    """
    resize = Resize(size=size)
    if isinstance(labels, np.ndarray):
        return resize.transform(x, labels=labels, num_class=num_class)
    return resize.transform(x)


class Resize(Image):
    """Apply resize transformation to the input x and labels.

    :param tuple size: size of ('Height', "Width")
    """

    def __init__(self, size=(0, 0)):
        super(Resize, self).__init__()
        self.size = size

    def transform(self, x, labels=None, num_class=0, **kwargs):
        """Performs a resize transformation of a Numpy Image x.
        if x is a Batch, apply Resize transform to Batch.
        if arguments include labels, apply label transformation.

        :param ndarray x: 3 or 4(batch) dimensional x
        :param ndarray labels: rectangle labels(2-dimensional array)
                               ex:) np.array([[center x, center y, x_top_left, height, 0, 0, 0, 1, 0]])
        :param int num_class: number of class of datasets
        :return: Images(4 dimension) of resize transformed. If including labels, return with transformed labels
        :rtype: ndarray

        :Example:
        >>> from renom.utility.image.data_augmentation.resize import Resize
        >>> from PIL import Image as im
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> image = im.open(image path) # ex:) "./image_folder/camera.jpg"
        >>> image = np.array(image).astype(np.float32)
        >>> rs = Resize(size=(300, 500))
        >>> resize_image = rs.transform(image)
        >>> fig, axes = plt.subplots(2, 1)
        >>> axes[0].imshow(image/255); axes[0].set_title("Original Image")
        >>> axes[1].imshow(resize_image[0] / 255); axes[1].set_title("Resize One Image")
        >>> plt.show()

        """
        if self.size == (0, 0):
            print("Please set size of Resized Images. Return Original Images")
            return x

        resized_images = None
        batch_x, batch_size, original_size = self.check_x_dim(x.copy())

        resized_images = np.zeros(
            (batch_x.shape[0], self.size[0], self.size[1], batch_x.shape[3]), dtype=batch_x.dtype)

        # Dealing with gray images
        if batch_x.shape[-1] == 1:
            for index, image in enumerate(batch_x.copy() / 255.):
                if with_cv2:
                    resized_images[index] = cv2.resize(image[:, :, 0], (self.size[1], self.size[0]))[
                        :, :, np.newaxis] * 255.
                else:
                    resized_images[index] = (
                        rs(image,
                            (self.size[0], self.size[1], batch_x.shape[3]),
                            mode="reflect") * 255).astype(batch_x.dtype)
        # RGB images
        else:
            for index, image in enumerate(batch_x.copy() / 255.):
                if with_cv2:
                    resized_images[index] = cv2.resize(image, (self.size[1], self.size[0])) * 255.
                else:
                    resized_images[index] = rs(
                        image, (self.size[0], self.size[1], batch_x.shape[3]), mode="reflect") * 255.

        if isinstance(labels, np.ndarray):
            return resized_images, self._labels_transform(labels, num_class, batch_x.shape[1:3])

        return resized_images

    def _labels_transform(self, labels, num_class, img_shape):
        """Perform labels transformation for rectangle. Calculate center x, y and width, height of rectangle
        """
        transform_labels = labels.copy()
        block_len = 4 + num_class
        num_block = labels[0].shape[0] // block_len

        # (height, width)
        img_center = ((img_shape[0]) / 2., (img_shape[1]) / 2.)
        for index, label in enumerate(transform_labels):
            for block in range(num_block):
                if label[block * (block_len):(block + 1) * (block_len)][2:4].all() == 0.:
                    continue
                center_x, center_y, label_width, label_height = label[block * (
                    block_len):(block + 1) * (block_len)][:4]
                label_height *= self.size[0] / float(img_shape[0])
                label_width *= self.size[1] / float(img_shape[1])

                if center_x <= img_center[1]:
                    center_x = (self.size[1]) / 2. - (img_center[1] -
                                                      center_x) * self.size[1] / float(img_shape[1])
                elif center_x > img_center[1]:
                    center_x = (self.size[1]) / 2. + (center_x - img_center[1]) * self.size[1] / float(
                        img_shape[1]) + (float(self.size[1]) / img_shape[1]) - 1

                if center_y <= img_center[0]:
                    center_y = (self.size[0]) / 2. - (img_center[0] -
                                                      center_y) * self.size[0] / float(img_shape[0])
                elif center_y > img_center[0]:
                    center_y = (self.size[0]) / 2. + (center_y - img_center[0]) * self.size[0] / float(
                        img_shape[0]) + (self.size[0] / float(img_shape[0])) - 1

                label[block * (block_len):(block + 1) * (block_len)
                      ][:4] = center_x, center_y, label_width, label_height
            transform_labels[index] = label
        return transform_labels
