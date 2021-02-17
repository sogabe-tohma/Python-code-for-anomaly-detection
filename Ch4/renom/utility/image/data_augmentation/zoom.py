from __future__ import division
import numpy as np
from scipy import ndimage
from renom.utility.image.data_augmentation.crop import crop
from renom.utility.image.data_augmentation.resize import resize
from renom.utility.image.data_augmentation.image import Image
from PIL import Image as im


def zoom(x, zoom_rate=(1, 1), random=False, labels=None, num_class=0):
    """Apply zoom in transformation to the input x.
    if x is a Batch, apply Zoom transform to Batch.
    needs zoom_rate (has to be > 1.0). If you use Random transformation, Zoom will be done
    randomly for values between the two limits given by the tuple or from 1 to zoom_rate.

    Args:
       x (ndarray): 3 or 4(batch) dimensional Images
       zoom_rate (tuple): zoom ratio. If use random transformation,
            zoom_rate can be an interval (lower bound > 1.0).
            If you want to specify the range, you could use tuple (min, max)
       random (bool): If True, apply random transformation.
       labels (ndarray): rectangle labels(2-dimensional array).
            ex:) np.array([[center x, center y, x_top_left, height, 0, 0, 0, 1, 0]])
       num_class (int): number of class of datasets (for rectangle transformation)

    Returns:
        (ndarray): Images(4 dimension) of zoom transformed. If including labels, return with transformed labels

    Example:
        >>> from renom.utility.image.data_augmentation.zoom import zoom
        >>> from PIL import Image as im
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> image = im.open(image path) # ex:) "./image_folder/camera.jpg"
        >>> image = np.array(image).astype(np.float32)
        >>> zoom_image = zoom(image, zooom_rate=2, random=True)
        >>> fig, axes = plt.subplots(2, 1)
        >>> axes[0].imshow(image/255); axes[0].set_title("Original Image")
        >>> axes[1].imshow(zoom_image[0] / 255); axes[1].set_title("Zoom One Image")
        >>> plt.show()
    """
    zoom = Zoom(zoom_rate=zoom_rate)
    return zoom.transform(x, random=random, labels=labels, num_class=num_class)


class Zoom(Image):
    """Apply zoom in transformation to the input x.

    Args:
        x (ndarray): 3 or 4(batch) dimensional Images
        zoom_rate (tuple): zoom ratio. If use random transformation,
            zoom_rate can be an interval (lower bound > 1.0).
            If you want to specify the range, you could use tuple (min, max)

    """

    def __init__(self, zoom_rate=(1, 1)):
        super(Zoom, self).__init__()
        self.zoom_rate = zoom_rate

    def transform(self, x, random=False, labels=None, num_class=0):
        """Performs a Zoom transformation of a Numpy Images.
        if x is a Batch, apply Zoom transform to Batch.
        needs zoom_rate (has to be > 1.0). If you use Random transformation, Zoom will be done
        randomly for values between the two limits given by the tuple or from 1 to zoom_rate.

        Args:
            x (ndarray): 3 or 4(batch) dimensional Image
            random (bool): If True, apply random transformation.
            labels (ndarray): rectangle labels(2-dimensional array).
                ex:) np.array([[center x, center y, x_top_left, height, 0, 0, 0, 1, 0]])
            num_class (int): number of class of datasets (for rectangle transformation)

        Returns:
            (ndarray): Images(4 dimension) of zoom transformed. If including labels, return with transformed labels

        Example:
            >>> from renom.utility.image.data_augmentation.zoom import Zoom
            >>> from PIL import Image as im
            >>> import matplotlib.pyplot as plt
            >>> import numpy as np
            >>> image = im.open(image path) # ex:) "./image_folder/camera.jpg"
            >>> image = np.array(image).astype(np.float32)
            >>> zo = Zoom(zoom_rate=2)
            >>> zoom_image = zo.transform(image, random=True)
            >>> fig, axes = plt.subplots(2, 1)
            >>> axes[0].imshow(image/255); axes[0].set_title("Original Image")
            >>> axes[1].imshow(zoom_image[0] / 255); axes[1].set_title("Zoom One Image")
            >>> plt.show()
        """
        zoom_images, batch_size, original_size = self.check_x_dim(x.copy())
        transformed_labels = None

        if random:
            if isinstance(self.zoom_rate, tuple):
                zoom_rate = np.random.uniform(
                    self.zoom_rate[0], self.zoom_rate[1], batch_size)
            else:
                zoom_rate = np.random.uniform(1.0, self.zoom_rate, batch_size)
            if labels is not None:
                transformed_labels = labels.copy()
                for index, (image, label, rate) in enumerate(zip(zoom_images, transformed_labels, zoom_rate)):
                    crop_size = (
                        int(float(original_size[0]) / rate), int(float(original_size[1]) / rate))
                    image, transformed_label = crop(
                        image, size=crop_size, random=True, labels=np.array([label]), num_class=num_class)
                    zoom_images[index], transformed_label = resize(
                        image, size=original_size, labels=transformed_label, num_class=num_class)
                    transformed_labels[index] = transformed_label[0]
            else:
                for index, (image, rate) in enumerate(zip(zoom_images, zoom_rate)):
                    crop_size = (
                        int(float(original_size[0]) / rate), int(float(original_size[1]) / rate))
                    image = crop(image, size=crop_size, random=True)
                    zoom_images[index] = resize(image, size=original_size)
        else:
            crop_size = (int(float(
                original_size[0]) / self.zoom_rate), int(float(original_size[1]) / self.zoom_rate))
            if labels is not None:
                zoom_images, transformed_labels = crop(
                    zoom_images, size=crop_size, labels=labels, num_class=num_class)
                zoom_images, transformed_labels = resize(
                    zoom_images, size=original_size, labels=transformed_labels, num_class=num_class)
            else:
                zoom_images = crop(zoom_images, size=crop_size)
                zoom_images = resize(zoom_images, size=original_size)

        if labels is not None:
            return zoom_images, transformed_labels
        else:
            return zoom_images
