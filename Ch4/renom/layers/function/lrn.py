#!/usr/bin/env python
# encoding:utf-8

from __future__ import division
import numpy as np
from renom.core import Node, to_value
import renom.cuda as cu
if cu.has_cuda():
    from renom.cuda.gpuvalue import get_gpu


class lrn(Node):

    def __new__(cls, x, n=5, k=2, a=1e-4, b=0.75):
        return cls.calc_value(x, n, k, a, b)

    @classmethod
    def _oper_cpu(cls, x, n, k, a, b):
        xs = np.square(x).view(np.ndarray)
        sum = np.array(xs.copy())
        sum = xs.copy()
        for i in range(1, n // 2 + 1):
            sum[:, i:, :, :] += xs[:, :-i, :, :]
            sum[:, :-i, :, :] += xs[:, i:, :, :]
        unit_scale = k + a * sum
        scale = unit_scale ** -b
        value = x * scale
        ret = cls._create_node(value)
        ret.attrs._x = x
        ret.attrs._n = n
        ret.attrs._a = a
        ret.attrs._b = b
        ret.attrs._unit_scale = unit_scale
        ret.attrs._scale = scale
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._x, Node):
            dy = to_value(dy)
            unit_scale = self.attrs._unit_scale
            scale = self.attrs._scale
            a = self.attrs._a
            b = self.attrs._b
            n = self.attrs._n
            x = self.attrs._x
            sum1 = (self * dy / unit_scale).view(np.ndarray)
            sum2 = sum1.copy()
            for i in range(1, n // 2 + 1):
                sum2[:, i:, :, :] += sum1[:, :-i, :, :]
                sum2[:, :-i, :, :] += sum1[:, i:, :, :]
            self.attrs._x._update_diff(context, dy * scale - 2 * a * b * x * sum2, **kwargs)

    @classmethod
    def _oper_gpu(cls, x, n, k, a, b):
        lrn_desc = cu.LRNDescriptor(n, a, b, k)
        y = get_gpu(x).empty_like_me()
        with cu.cudnn_handler() as handle:
            cu.cuLocalResponseNormalizationForward(handle, lrn_desc, get_gpu(x), get_gpu(y))
        ret = cls._create_node(y)
        ret.attrs._x = x
        ret.attrs._lrn_desc = lrn_desc
        return ret

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._x, Node):
            dx = get_gpu(self).empty_like_me()
            with cu.cudnn_handler() as handle:
                cu.cuLocalResponseNormalizationBackward(
                    handle, self.attrs._lrn_desc, get_gpu(self.attrs._x), get_gpu(self), dx, get_gpu(dy))
            self.attrs._x._update_diff(context, dx, **kwargs)


class Lrn:
    '''Local response normalization function [lrn]_ .

    .. math::
        y_{c_{out},j,i}= \\frac{x_{c_{in},i,j}}{(k + a{\sum_{c=max(0, i-n/2)}^{min(N-1, i+n/2)} (x_{c,i,j})})^b}

    :math:`x_{c,i,j}` represents the c th conv filter’s output at the position of (i, j) in the feature map.
    :math:`y_{c_{out},j,i}` represents the output of local response normalization.
    :math:`N` is the number of the feature maps.
    :math:`n` is the adjacent feature map number.
    The default argument values are recommended.

    Args:
        n (int): Number of images used to be normalized.
        k (int): Constant.
        a (float): Scale factor.
        b (float): Exponential factor.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> x = np.random.rand(3, 3, 32, 32)
        >>> layer = rm.Lrn()
        >>> z=layer(x)
        >>> z.shape
        (3, 3, 32, 32)

    .. [lrn] Alex Krizhevsky Krizhevsky, , Ilya IlyaSutskever Sutskever, Geoff.
        ImageNet Classification with Deep Convolutional Neural Networks

    '''

    def __init__(self, n=5, k=2, a=1e-4, b=0.75):
        self._n = n
        self._k = k
        self._a = a
        self._b = b

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return lrn(x, self._n, self._k, self._a, self._b)
