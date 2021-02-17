# encoding:utf-8
import numpy as np


class Image(object):
    "Parent Class of Image module."

    def __init__(self):
        pass

    def transform(self, *args, **kwargs):
        raise NotImplementedError

    def check_x_dim(self, x):
        """Check input dimension. If x dimension is 3, convert to 4 dimension by np.expand_dims(x, axis=0)
        Args:
            x (ndarray): Input matrix.
        """
        if x.ndim == 3:
            x = np.expand_dims(x, axis=0)
        if x.ndim != 4:
            raise NotImplementedError(
                "Input image should be numpy array of dim 3 (single image) or 4 (batch)")
        original_size = x.shape[1:3]
        batch_size = x.shape[0]
        return x, batch_size, original_size
