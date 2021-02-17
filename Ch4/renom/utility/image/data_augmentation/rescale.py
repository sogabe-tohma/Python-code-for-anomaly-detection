from __future__ import division
import numpy as np
from renom.utility.image.data_augmentation.image import Image


def rescale(x, option="", labels=None, **kwargs):
    """Performs a rescale transform of a Numpy images.
    if x is a Batch, apply rescale transform to Batch.

    Args:
        x (ndarray): 4(batch) dimensional images
        option (str): option of rescale.
            "zero": rescale images to [-0.5, 0.5].
            "vgg" : substract averate values of vgg datasets.
            other : rescale [0.0, 1.0].
        labels (ndarray): rectangle labels(2-dimensional array).
            ex:) np.array([[center x, center y, x_top_left, height, 0, 0, 0, 1, 0]])
    Returns:
        (ndarray): Images(4 dimension) of rescale transformed. If including labels, return with transformed labels

    :Example:
        >>> from rescale import rescale
        >>> from PIL import Image as im
        >>> import numpy as np
        >>> image = im.open(image path) # ex:) "./image_folder/camera.jpg"
        >>> image = np.array(image).astype(np.float32)
        >>> rescale_image = rescale(image, option="zero")
        >>> print(rescale_image.min(), rescale_image.max())
        (-0.5, 0.5)
    """
    rescale = Rescale(option=option)
    if isinstance(labels, np.ndarray):
        return rescale.transform(x, labels=labels)
    return rescale.transform(x)


class Rescale(Image):
    """Apply rescale transformation to the input x

    Args:
        option (str): option of rescale.
                       "zero": rescale images to [-0.5, 0.5].
                       "vgg" : substract averate values of vgg datasets.
                       other : rescale [0.0, 1.0].
    """

    def __init__(self, option=""):
        super(Rescale, self).__init__()
        self.option = option

    def transform(self, x, labels=None, **kwargs):
        """Performs a rescale transform of a Numpy images.
        if x is a Batch, apply rescale transform to Batch

        Args:
            x (ndarray): 3 or 4(batch) dimensional images
            labels (ndarray): rectangle labels(2-dimensional array)
                               ex:) np.array([[center x, center y, x_top_left, height, 0, 0, 0, 1, 0]])
        Returns:
            (ndarray): Images(4 dimension) of rescale transformed. If including labels, return with transformed labels

        >>> from renom.utility.image.data_augmentation.rescale import Rescale
        >>> from PIL import Image as im
        >>> import numpy as np
        >>> image = im.open(image path) # ex:) "./image_folder/camera.jpg"
        >>> image = np.array(image).astype(np.float32)
        >>> rs = Rescale(option="zero")
        >>> rescale_image = rs.transform(image)
        >>> print(rescale_image.min(), rescale_image.max())
        (-0.5, 0.5)
        """
        rescale_images, batch_size, original_size = self.check_x_dim(x.copy())
        rescale_images = self._rescale_images(rescale_images)

        if isinstance(labels, np.ndarray):
            return rescale_images, labels.copy()
        return rescale_images

    def _rescale_images(self, x):
        if self.option == "vgg":
            x = self._vgg_preprocess(x)
        elif self.option == "zero":
            x = x / 255. - 0.5
        elif isinstance(self.option, list):
            x = x / 255. * (self.option[1] - self.option[0]) + self.option[0]
        else:
            x = x / 255.
        return x

    def _vgg_preprocess(self, x):
        x[:, :, :, 0] -= 123.68
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 103.939
        return x
