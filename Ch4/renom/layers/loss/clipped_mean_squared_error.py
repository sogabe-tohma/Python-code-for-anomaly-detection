#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import numpy as np
from renom.core import BinOp, Node
import renom.cuda as cu
if cu.has_cuda():
    from renom.cuda.gpuvalue import get_gpu
from renom.layers.function.utils import tuplize


class clipped_mean_squared_error(Node):

    def __new__(cls, lhs, rhs, clip=1, reduce_sum=True):
        assert rhs.ndim > 1, "Input arrays must have no less than 2 dimension."
        c = clip if isinstance(clip, tuple) else (-clip, clip)
        return cls.calc_value(lhs, rhs, c, reduce_sum)

    @classmethod
    def _oper_cpu(cls, lhs, rhs, clip, reduce_sum):
        N = len(lhs)
        if reduce_sum:
            z = np.sum((lhs - rhs) ** 2) / (N * 2)
        else:
            z = (lhs - rhs) ** 2 / (N * 2)
        ret = cls._create_node(z)
        ret.attrs._lhs = lhs
        ret.attrs._rhs = rhs
        ret.attrs._clip = clip
        return ret

    @classmethod
    def _oper_gpu(cls, lhs, rhs, clip, reduce_sum):
        N = len(lhs)
        if reduce_sum:
            z = cu.cusum(get_gpu((get_gpu(lhs) - get_gpu(rhs)) ** 2)) / (N * 2)
        else:
            z = get_gpu((get_gpu(lhs) - get_gpu(rhs)) ** 2) / (N * 2)
        ret = cls._create_node(z)
        ret.attrs._lhs = lhs
        ret.attrs._rhs = rhs
        ret.attrs._clip = clip
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        sub = self.attrs._lhs - self.attrs._rhs
        if isinstance(self.attrs._lhs, Node):
            N = len(self.attrs._lhs)
            clip = self.attrs._clip
            dx = np.clip(sub * dy, clip[0], clip[1])
            self.attrs._lhs._update_diff(context, dx / N, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._lhs, Node):
            N = len(self.attrs._lhs)
            clip = self.attrs._clip
            sub = get_gpu(self.attrs._lhs) - get_gpu(self.attrs._rhs)
            dx = sub * get_gpu(dy)
            cu.cumin(clip[1], dx, dx)
            cu.cumax(clip[0], dx, dx)
            self.attrs._lhs._update_diff(context, dx / N, **kwargs)


class ClippedMeanSquaredError:
    """Cliped mean squared error function.
    In the forward propagation, this function
    yields same calculation as mean squared error.

    In the backward propagation, this function
    calculates following formula.

    .. math::
        \\frac{dE}{dx}_{clipped} = max(min(\\frac{dE}{dx}, clip), -clip)

    Args:
        x (ndarray,Node): Input data.
        y (ndarray,Node): Target data.
        clip (float,tuple): Clipping threshold.
        reduce_sum (bool): If True is given, the result array will be summed up and returns scalar value.

    Returns:
        (Node, ndarray): Clipping mean squared error.

    Example:
        >>> import renom as rm
        >>> import numpy as np
        >>>
        >>> x = np.array([[1, 1]])
        >>> y = np.array([[-1, -1]])
        >>> print(x.shape, y.shape)
        ((1, 2), (1, 2))
        >>> loss = rm.clipped_mean_squared_error(x, y)
        >>> print(loss)
        clipped_mean_squared_error(4.0)
        >>>
        >>> # Also you can call this function with alias.
        >>> loss = rm.cmse(x, y)
        >>> print(loss)
        clipped_mean_squared_error(4.0)


    Raises:
        AssertionError: An assertion error will be raised if the given tensor dimension is less than 2.

    """

    def __init__(self, clip=1.0, reduce_sum=True):
        c = clip if isinstance(clip, tuple) else (-clip, clip)
        self._clip = c
        self.reduce_sum = reduce_sum

    def __call__(self, x, y):
        return clipped_mean_squared_error(x, y, self._clip, self.reduce_sum)
