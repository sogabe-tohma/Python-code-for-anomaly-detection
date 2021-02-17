from __future__ import division
import numpy as np
from renom.utility.image.data_augmentation.image import Image


def flip(x, flip=0, random=False, labels=None, num_class=0):
    """Performs a flip of a Numpy images.
    if x is a Batch, apply flip transform to Batch.
    if arguments include labels, apply label transformation.

    Args:
        x (ndarray): 3 or 4(batch) dimensional images
        flip (int): 1 means Horizontal Flip.
                     2 means Vertical Flip.
                     3 means both.
                     else no flip conversion.
        random (bool): apply random flip or not
        labels (ndarray): rectangle labels(2-dimensional array)
                           ex:) np.array([[center x, center y, x_top_left, height, 0, 0, 0, 1, 0]])
        num_class (int): number of class of datasets (for rectangle transformation)

    Returns:
        (ndarray): Images(4 dimension) of flip transformed. If including labels, return with transformed labels

    Example:
        >>> from renom.utility.image.data_augmentation.flip import flip
        >>> from PIL import Image as im
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> image = im.open(image path) # ex:) "./image_folder/camera.jpg"
        >>> image = np.array(image).astype(np.float32)
        >>> flip_image = flip(image, flip=1) # Horizontal Flip
        >>> fig, axes = plt.subplots(2, 1)
        >>> axes[0].imshow(image/255); axes[0].set_title("Original Image")
        >>> axes[1].imshow(flip_image[0] / 255); axes[1].set_title("Flip One Image")
        >>> plt.show()
    """
    flip = Flip(flip)
    if isinstance(labels, np.ndarray):
        return flip.transform(x, random=random, labels=labels, num_class=num_class)
    return flip.transform(x, random=random)


class Flip(Image):
    """Apply flip transformation to the input x and labels.

    Args:
        flip (int): 1 means Horizontal Flip.
                     2 means Vertical Flip.
                     3 means both.
                     else no flip conversion.
    """

    def __init__(self, flip):
        super(Flip, self).__init__()
        self.flip = flip

    def transform(self, x, random=False, labels=None, num_class=0):
        """Performs a flip of a Numpy images.
        if x is a Batch, apply flip transform to Batch
        if arguments include labels, apply label transformation

        Args:
            x (ndarray): 3 or 4(batch) dimensional images
            random (bool): apply random flip or not
            labels (ndarray): rectangle labels(2-dimensional array)
                               ex:) np.array([[center x, center y, x_top_left, height, 0, 0, 0, 1, 0]])
            num_class (int): number of class of datasets (for rectangle transformation)

        Returns:
            (ndarray): Images(4 dimension) of flip transformed. If including labels, return with transformed labels

        :Example:
            >>> from renom.utility.image.data_augmentation.flip import Flip
            >>> from PIL import Image as im
            >>> import matplotlib.pyplot as plt
            >>> import numpy as np
            >>> image = im.open(image path) # ex:) "./image_folder/camera.jpg"
            >>> image = np.array(image).astype(np.float32)
            >>> fl = Flip(flip=1) # Horizontal Flip
            >>> flip_image = fl.transform(image)
            >>> fig, axes = plt.subplots(2, 1)
            >>> axes[0].imshow(image/255); axes[0].set_title("Original Image")
            >>> axes[1].imshow(flip_image[0] / 255); axes[1].set_title("Flip One Image")
            >>> plt.show()
        """
        flipped_images, batch_size, original_size = self.check_x_dim(x.copy())
        shuffle = None
        transformed_labels = None
        if random:
            shuffle = np.random.choice(x.shape[0], int(
                np.ceil(x.shape[0] / 2.)), replace=False)
            flipped_images[shuffle] = [self._get_fliped_image(
                flip_image) for flip_image in flipped_images[shuffle]]
            if isinstance(labels, np.ndarray):
                transformed_labels = self._labels_transform(
                    labels, num_class, shuffle, x.shape[1:3])
        else:
            flipped_images[:] = [self._get_fliped_image(
                flip_image) for flip_image in flipped_images]
            if isinstance(labels, np.ndarray):
                transformed_labels = self._labels_transform(
                    labels, num_class, np.arange(x.shape[0]), x.shape[1:3])

        if isinstance(labels, np.ndarray):
            return flipped_images, transformed_labels
        return flipped_images

    def _labels_transform(self, labels, num_class, shuffle, img_shape):
        transformed_labels = labels.copy()
        block_len = 4 + num_class
        num_block = labels[0].shape[0] // block_len
        for index, label in zip(shuffle, transformed_labels[shuffle]):
            for block in range(num_block):
                if label[block * (block_len):(block + 1) * (block_len)][2:4].all() == 0.:
                    break
                if self.flip == 1:
                    label[block * (block_len):(block + 1) * (block_len)][0] = img_shape[1] - \
                        label[block * (block_len):(block + 1) * (block_len)][0] - 1
                elif self.flip == 2:
                    label[block * (block_len):(block + 1) * (block_len)][1] = img_shape[0] - \
                        label[block * (block_len):(block + 1) * (block_len)][1] - 1
                elif self.flip == 3:
                    center_x, center_y = label[block * (block_len):(block + 1) * (block_len)][:2]
                    label[block * (block_len):(block + 1) * (block_len)
                          ][:2] = img_shape[1] - center_x - 1, img_shape[0] - center_y - 1
                else:
                    return transformed_labels
            transformed_labels[index] = label
        return transformed_labels

    def _get_fliped_image(self, image):
        if self.flip == 1:
            return image[:, ::-1]
        elif self.flip == 2:
            return image[::-1, :]
        elif self.flip == 3:
            return image[::-1, ::-1]
        else:
            return image
