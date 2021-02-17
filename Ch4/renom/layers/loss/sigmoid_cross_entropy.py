#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import numpy as np
from renom.core import Node, to_value
import renom.cuda as cu
if cu.has_cuda():
    from renom.cuda.gpuvalue import get_gpu


class sigmoid_cross_entropy(Node):

    def __new__(cls, lhs, rhs, reduce_sum=True):
        assert rhs.ndim > 1, "Input arrays must have no less than 2 dimension."
        return cls.calc_value(lhs, rhs, reduce_sum=reduce_sum)

    @classmethod
    def _oper_cpu(cls, lhs, rhs, reduce_sum):
        N = len(lhs)
        z = 1. / (1. + np.exp(to_value(-lhs)))
        if reduce_sum:
            loss = -np.sum(to_value(rhs) * np.log(z + 1e-8) +
                           to_value(1 - rhs) * np.log(1 - z + 1e-8)) / N
        else:
            loss = -(to_value(rhs) * np.log(z + 1e-8) +
                     to_value(1 - rhs) * np.log(1 - z + 1e-8)) / N

        ret = cls._create_node(loss)
        ret.attrs._z = z
        ret.attrs._lhs = lhs
        ret.attrs._rhs = rhs
        return ret

    @classmethod
    def _oper_gpu(cls, lhs, rhs, reduce_sum):
        N = len(lhs)
        z = get_gpu(lhs).empty_like_me()
        tmp1 = get_gpu(lhs).empty_like_me()
        tmp2 = get_gpu(lhs).empty_like_me()
        cu.cusigmoid(get_gpu(lhs), z)
        cu.cucross_entropy(get_gpu(z), get_gpu(rhs), tmp1)
        cu.cucross_entropy(get_gpu(-z + 1.), get_gpu(-rhs + 1.), tmp2)
        if reduce_sum:
            loss = cu.cusum(-(tmp1 + tmp2)) / N
        else:
            loss = -(tmp1 + tmp2) / N
        ret = cls._create_node(loss)
        ret.attrs._z = z
        ret.attrs._lhs = lhs
        ret.attrs._rhs = rhs
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._lhs, Node):
            sub = self.attrs._z - self.attrs._rhs
            N = len(self.attrs._z)
            self.attrs._lhs._update_diff(context, sub * dy / N, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._lhs, Node):
            N = len(self.attrs._lhs)
            sub = get_gpu(self.attrs._z) - get_gpu(self.attrs._rhs)
            self.attrs._lhs._update_diff(context, sub * get_gpu(dy) / N, **kwargs)


class SigmoidCrossEntropy:
    """This function evaluates the loss between target ``y`` and output
    of sigmoid activation ``z`` using cross entropy.

    .. math::
        z_{nk} &= \\frac{1}{1 + \exp(-x_{nk})} \\\\
        E(x) &= -\\frac{1}{N}\sum_{n}^{N}\sum_{k}^{K}y_{nk}\log(z_{nk})+(1-y_{nk})\log(1-z_{nk})

    Args:
        x (ndarray,Node): Input array.
        y (ndarray,Node): Target array.
        reduce_sum (bool): If True is given, the result array will be summed up and returns scalar value.

    Returns:
        (Node, ndarray): Cross entropy error between sigmoid(x) and target y.

    Example:
        >>> import renom as rm
        >>> import numpy as np
        >>>
        >>> x = np.array([[0, 1]])
        >>> y = np.array([[1, 1]])
        >>> loss_func = rm/SigmoidCrossEntropy()
        >>> loss = loss_func(x, y)
        >>> print(loss)
        1.0064088106155396
        >>>
        >>> loss = rm.sigmoid_cross_entropy(x, y)
        >>> print(loss)
        1.0064088106155396
        >>>
        >>> # You can also call this function with alias.
        >>> loss = rm.sgce(x, y)
        >>> print(loss)
        1.0064088106155396


    Raises:
        AssertionError: An assertion error will be raised if the given tensor dimension is less than 2.

    """

    def __call__(self, lhs, rhs, reduce_sum=True):
        return sigmoid_cross_entropy(lhs, rhs, reduce_sum=reduce_sum)
