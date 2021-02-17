#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from renom.core import Node
from renom.layers.function.utils import im2col, col2im, out_size, tuplize
import renom.cuda as cu
if cu.has_cuda():
    from renom.cuda.gpuvalue import GPUValue, get_gpu


class pool_base(Node):

    def __new__(cls, x, filter=3, stride=1, padding=0, ceil_mode=False):
        filter, stride, padding = (tuplize(x) for x in (filter, stride, padding))
        in_shape = x.shape[1:]
        out_shape = [x.shape[1], ]
        out_shape.extend(out_size(x.shape[2:], filter, stride, padding, ceil_mode=ceil_mode))
        return cls.calc_value(x, in_shape, out_shape, filter, stride, padding)

    def _backward_gpu(self, context, dy, **kwargs):
        dx = get_gpu(self.attrs._x).empty_like_me()
        with cu.cudnn_handler() as handle:
            cu.cuPoolingBackward(handle, self.attrs._pool_desc, get_gpu(
                self.attrs._x), get_gpu(self), get_gpu(dy), dx)
        if isinstance(self.attrs._x, Node):
            self.attrs._x._update_diff(context, dx, **kwargs)


class max_pool2d(pool_base):

    @classmethod
    def _oper_cpu(cls, x, in_shape, out_shape, karnel, stride, padding):
        col = im2col(x, out_shape[1:], karnel,
                     stride, padding)
        n, ic, kh, kw, oh, ow = col.shape
        col = col.reshape(n, ic, kh * kw, oh, ow)
        index = np.argmax(col, axis=2)
        value = np.max(col, axis=2)
        ret = cls._create_node(value)
        ret.attrs._index = index
        ret.attrs._x = x
        ret.attrs._in_shape = in_shape
        ret.attrs._out_shape = out_shape
        ret.attrs._kernel = karnel
        ret.attrs._stride = stride
        ret.attrs._padding = padding
        return ret

    @classmethod
    def _oper_gpu(cls, x, in_shape, out_shape, karnel, stride, padding):
        N = x.shape[0]
        pool_desc = cu.PoolingDescriptor(karnel, padding, stride, pool_mode=0)
        _x = get_gpu(x)
        y = GPUValue(shape=tuple([N, ] + list(out_shape)))
        with cu.cudnn_handler() as handle:
            cu.cuPoolingForward(handle, pool_desc, _x, y)
        ret = cls._create_node(y)
        ret.attrs._pool_desc = pool_desc
        ret.attrs._kernel = karnel
        ret.attrs._stride = stride
        ret.attrs._padding = padding
        ret.attrs._x = x
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._x, Node):
            N = len(dy)
            index = self.attrs._index
            col = np.zeros((N, self.attrs._in_shape[0], self.attrs._kernel[0],
                            self.attrs._kernel[1], self.attrs._out_shape[1], self.attrs._out_shape[2]))
            col_k = np.rollaxis(col.reshape(
                N, self.attrs._in_shape[0], -1, self.attrs._out_shape[1], self.attrs._out_shape[2]), 2)
            for i in np.ndindex(N, self.attrs._in_shape[0], self.attrs._out_shape[1], self.attrs._out_shape[2]):
                col_k[index[i]][i] = dy[i]
            dx = col2im(col, self.attrs._in_shape[1:], self.attrs._stride, self.attrs._padding)
            self.attrs._x._update_diff(context, dx, **kwargs)


class average_pool2d(pool_base):

    @classmethod
    def _oper_cpu(cls, x, in_shape, out_shape, karnel, stride, padding):
        col = im2col(x, out_shape[1:], karnel,
                     stride, padding)
        n, ic, kh, kw, oh, ow = col.shape
        col = col.reshape(n, ic, kh * kw, oh, ow)
        value = np.mean(col, axis=2)
        ret = cls._create_node(value)
        ret.attrs._x = x
        ret.attrs._in_shape = in_shape
        ret.attrs._out_shape = out_shape
        ret.attrs._kernel = karnel
        ret.attrs._stride = stride
        ret.attrs._padding = padding
        return ret

    @classmethod
    def _oper_gpu(cls, x, in_shape, out_shape, karnel, stride, padding):
        N = x.shape[0]
        pool_desc = cu.PoolingDescriptor(karnel, padding, stride, pool_mode=1)
        y = GPUValue(shape=tuple([N, ] + list(out_shape)))
        with cu.cudnn_handler() as handle:
            cu.cuPoolingForward(handle, pool_desc, get_gpu(x), y)
        ret = cls._create_node(y)
        ret.attrs._pool_desc = pool_desc
        ret.attrs._kernel = karnel
        ret.attrs._stride = stride
        ret.attrs._padding = padding
        ret.attrs._x = x
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._x, Node):
            N = len(dy)
            col = np.zeros((N, self.attrs._in_shape[0], self.attrs._kernel[0],
                            self.attrs._kernel[1], self.attrs._out_shape[1], self.attrs._out_shape[2]))
            col_k = np.rollaxis(col.reshape(
                N, self.attrs._in_shape[0], -1, self.attrs._out_shape[1], self.attrs._out_shape[2]), 2)
            col_k[:] = dy / float(len(col_k))
            dx = col2im(col, self.attrs._in_shape[1:], self.attrs._stride, self.attrs._padding)
            self.attrs._x._update_diff(context, dx, **kwargs)


class PoolBase(object):

    def __init__(self, filter=3,
                 padding=0, stride=1):
        self._padding, self._stride, self._kernel = (tuplize(x) for x in (padding, stride, filter))

    def __call__(self, x):
        assert len(x.shape) == 4, "The dimension of input array must be 4. Actual dim is {}".format(x.ndim)
        assert all([s > 0 for s in x.shape[2:]]), \
            "The shape of input array {} is too small. Please give an array which size is lager than 0.".format(
                x.shape)
        return self.forward(x)


class MaxPool2d(PoolBase):
    '''Max pooling function.
    In the case of int input, filter, padding, and stride, the shape will be symmetric.

    Args:
        filter (tuple,int): Filter size of the convolution kernel.
        padding (tuple,int): Size of the zero-padding around the image.
        stride (tuple,int): Stride-size of the convolution.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>>
        >>> x = np.random.rand(3, 3, 32, 32)
        >>> layer = rm.MaxPool2d(filter=3)
        >>> z = layer(x)
        >>> z.shape
        (3, 3, 30, 30)
        >>> z = rm.max_pool2d(x, filter=(3, 3), stride=(1, 1), padding=(0,0))
        >>> z.shape
        (3, 3, 30, 30)

    '''

    def forward(self, x):
        return max_pool2d(x, self._kernel, self._stride, self._padding)


class AveragePool2d(PoolBase):
    '''Average pooling function.
    In the case of int input, filter, padding, and stride, the shape will be symmetric.

    Args:
        filter (tuple,int): Filter size of the convolution kernel.
        padding (tuple,int): Size of the zero-padding around the image.
        stride (tuple,int): Stride-size of the convolution.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>>
        >>> x = np.random.rand(3, 3, 32, 32)
        >>> layer = rm.AveragePool2d(filter=3)
        >>> z = layer(x)
        >>> z.shape
        (3, 3, 30, 30)
        >>> z = rm.average_pool2d(x, filter=(3, 3), stride=(1, 1), padding=(0,0))
        >>> z.shape
        (3, 3, 30, 30)

    '''

    def forward(self, x):
        return average_pool2d(x, self._kernel, self._stride, self._padding)
