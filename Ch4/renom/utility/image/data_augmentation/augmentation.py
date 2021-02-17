import numpy as np
from renom.utility.image.data_augmentation.image import Image


class DataAugmentation(Image):
    """Apply transformation to the input x and labels.
    You could choose transform function from below.
    ["Flip", "Resize", "Crop", "Color_jitter", "Rescale", "Rotate", "Shift"].

    Args:
        converter_list (list): list of instance for converter.
        random (bool): apply random transformation or not
    """

    def __init__(self, converter_list, random=False):
        super(DataAugmentation, self).__init__()
        self.converter_list = converter_list
        self.random = random

    def create(self, x, labels=None, num_class=0):
        """Performs a DataAugmentation of a Numpy images.
        if x is a Batch, apply DataAugmentation to Batch.
        if arguments include labels, apply label transformation.

        Args:
            x (ndarray): 3 or 4(batch) dimensional images. dtype is float32. value=[0.0, 255.0].
            labels (ndarray): labels for classification, detection and segmentation. 2-dimensional array
            num_class (int): number of class of datasets

        Returns:
            (ndarray): Images(4 dimension) of augment transformed. If including labels, return with transformed labels

        Example:
            >>> import matplotlib.pyplot as plt
            >>> from PIL import Image as im
            >>> from renom.utility.image.data_augmentation import *
            >>> image = im.open("/Users/tsujiyuuki/env_python/code/my_code/Utilities/doc/img_autodoc/2007_000027.jpg")
            >>> image = np.array(image, dtype=np.float32)
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
            >>> augment_image = datagenerator.create(image)
            >>> fig, axes = plt.subplots(2, 1)
            >>> axes[0].imshow(image/255); axes[0].set_title("Original Image")
            >>> axes[1].imshow(augment_image[0] / 255); axes[1].set_title("Shift One Image")
            >>> plt.show()

        """
        augmented_images = x.copy().astype(np.float32)
        if isinstance(labels, np.ndarray):
            transformed_labels = labels.copy()
            for converter in self.converter_list:
                augmented_images, transformed_labels = converter.transform(
                    augmented_images, random=self.random, labels=transformed_labels, num_class=num_class)
            return augmented_images, transformed_labels

        for converter in self.converter_list:
            augmented_images = converter.transform(
                augmented_images, random=self.random, labels=labels, num_class=num_class)
        return augmented_images
