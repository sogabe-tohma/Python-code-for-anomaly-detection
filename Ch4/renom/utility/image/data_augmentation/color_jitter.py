from __future__ import division
import numpy as np
from renom.utility.image.data_augmentation.image import Image
try:
    import cv2
    with_cv2 = True
except Exception:
    from skimage import color as cl
    with_cv2 = False


def color_jitter(x, h=None, s=None, v=None, random=False, labels=None, **kwargs):
    """Performs a HSV color jitter of a RGB images.
    if x is a Batch, apply jitter transform to Batch.
    if arguments include labels, apply label transformation.

    Args:
        x (ndarray): 3 or 4(batch) dimensional RGB images
        h (tuple): multiple value to h channel of HSV color space.
                    when you apply random transformation, please use tuple (min h, max h)
        s (tuple): multiple value to s channel of HSV color space.
                    when you apply random transformation, please use tuple (min s, max s)
        v (tuple): multiple value to v channel of HSV color space.
                    when you apply random transformation, please use tuple (min v, max v)
        random (bool): If True, apply random jitter transform
        labels (ndarray): rectangle labels(2-dimensional array)
                           ex:) np.array([[center x, center y, x_top_left, height, 0, 0, 0, 1, 0]])
    Returns:
        (ndarray): Images(4 dimension) of jitter transformed. If including labels, return with transformed labels

    Example:
        >>> from renom.utility.image.data_augmentation.color_jitter import color_jitter
        >>> from PIL import Image as im
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> image = im.open(image path) # ex:) "./image_folder/camera.jpg"
        >>> image = np.array(image).astype(np.float32)
        >>> jitter_image = color_jitter(image, v=2)
        >>> fig, axes = plt.subplots(2, 1)
        >>> axes[0].imshow(image/255); axes[0].set_title("Original Image")
        >>> axes[1].imshow(jitter_image[0] / 255); axes[1].set_title("Jitter One Image")
        >>> plt.show()

    """
    cj = ColorJitter(h=h, s=s, v=v)
    if isinstance(labels, np.ndarray):
        return cj.transform(x, random=random, labels=labels)
    return cj.transform(x, random=random)


class ColorJitter(Image):
    """Apply color jitter transformation to the input x and labels.

    Args:
        h (tuple): multiple value to h channel of HSV color space.
                    when you apply random transformation, please use tuple (min h, max h)
        s (tuple): multiple value to s channel of HSV color space.
                    when you apply random transformation, please use tuple (min s, max s)
        v (tuple): multiple value to v channel of HSV color space.
                    when you apply random transformation, please use tuple (min v, max v)
    """

    def __init__(self, h=None, s=None, v=None):
        super(ColorJitter, self).__init__()
        self.h = h
        self.s = s
        self.v = v

    def transform(self, x, random=False, labels=None, **kwargs):
        """Performs a HSV color jitter of a RGB images.
        if x is a Batch, apply jitter transform to Batch
        if arguments include labels, apply label transformation

        Args:
            x (ndarray): 3 or 4(batch) dimensional RGB images
            random (bool): apply random jitter or not
            labels (ndarray): rectangle labels(2-dimensional array)
                               ex:) np.array([[center x, center y, x_top_left, height, 0, 0, 0, 1, 0]])
        Returns:
            (ndarray): Images(4 dimension) of jitter transformed. If including labels, return with transformed labels

        Example:
            >>> from renom.utility.image.data_augmentation.color_jitter import ColorJitter
            >>> from PIL import Image as im
            >>> import matplotlib.pyplot as plt
            >>> import numpy as np
            >>> image = im.open(image path) # ex:) "./image_folder/camera.jpg"
            >>> image = np.array(image).astype(np.float32)
            >>> cj = ColorJitter(v=2)
            >>> jitter_image = cj.transform(image)
            >>> fig, axes = plt.subplots(2, 1)
            >>> axes[0].imshow(image/255); axes[0].set_title("Original Image")
            >>> axes[1].imshow(jitter_image[0] / 255); axes[1].set_title("Jitter One Image")
            >>> plt.show()

        """
        jittered_images, batch_size, original_size = self.check_x_dim(x.copy())
        # Only processes RGB images, if gray then skip
        if jittered_images.shape[-1] == 3:
            jittered_images[:] = [self._color_jitter(
                jitter_image, random=random) for jitter_image in jittered_images]

        if isinstance(labels, np.ndarray):
            return jittered_images, labels.copy()
        return jittered_images

    def _color_jitter(self, x, random=False):
        if with_cv2:
            hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
        else:
            hsv = cl.rgb2hsv(x.astype(np.float32) / 255.)  # x.max())

        h, s, v = self.h, self.s, self.v
        if random:
            if h:
                if isinstance(h, tuple):
                    h = np.random.uniform(h[0], h[1])
                else:
                    h = np.random.uniform(1.0, h)
            if s:
                if isinstance(s, tuple):
                    s = np.random.uniform(s[0], s[1])
                else:
                    s = np.random.uniform(1.0, s)
            if v:
                if isinstance(v, tuple):
                    v = np.random.uniform(v[0], v[1])
                else:
                    v = np.random.uniform(1.0, v)
        max_values = [359, 1, 255] if with_cv2 else [1.0, 1.0, 1.0]
        if h:
            bool_image = hsv[:, :, 0] * h >= max_values[0]
            hsv[:, :, 0][bool_image] = max_values[0]
            hsv[:, :, 0][~bool_image] = hsv[:, :, 0][~bool_image] * h
        if s:
            bool_image = hsv[:, :, 1] * s >= max_values[1]
            hsv[:, :, 1][bool_image] = max_values[1]
            hsv[:, :, 1][~bool_image] = hsv[:, :, 1][~bool_image] * s
        if v:
            bool_image = hsv[:, :, 2] * v >= max_values[2]
            hsv[:, :, 2][bool_image] = max_values[2]
            hsv[:, :, 2][~bool_image] = hsv[:, :, 2][~bool_image] * v

        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) if with_cv2 else (
            cl.hsv2rgb(hsv) * 255).astype(np.float32)
        return rgb
