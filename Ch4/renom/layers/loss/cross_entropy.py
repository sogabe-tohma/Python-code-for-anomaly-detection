#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import numpy as np
from renom.core import Node, to_value
import renom.cuda as cu
if cu.has_cuda():
    from renom.cuda.gpuvalue import get_gpu
from renom.operation import log


class cross_entropy(Node):

    def __new__(cls, lhs, rhs, reduce_sum=True):
        assert rhs.ndim > 1, "Input arrays must have no less than 2 dimension."
        return cls.calc_value(lhs, rhs, reduce_sum=reduce_sum)

    @classmethod
    def _oper_cpu(cls, lhs, rhs, reduce_sum):
        log_lhs = np.log(lhs + 1e-8)
        if reduce_sum:
            ret = cls._create_node(-np.sum(rhs * log_lhs))
        else:
            ret = cls._create_node(-rhs * log_lhs)
        ret.attrs._log_lhs = log_lhs
        ret.attrs._rhs = rhs
        ret.attrs._lhs = lhs
        return ret

    @classmethod
    def _oper_gpu(cls, lhs, rhs, reduce_sum):
        log_lhs = log(lhs + 1e-8)
        if reduce_sum:
            ret = cls._create_node(-cu.cusum(get_gpu(log_lhs * rhs)))
        else:
            ret = cls._create_node(-get_gpu(log_lhs * rhs))
        ret.attrs._log_lhs = log_lhs
        ret.attrs._rhs = rhs
        ret.attrs._lhs = lhs
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._rhs, Node):
            self.attrs._rhs._update_diff(context, -dy * self.attrs._log_lhs, **kwargs)

        if isinstance(self.attrs._lhs, Node):
            self.attrs._lhs._update_diff(context, -dy * self.attrs._rhs / self.attrs._lhs, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._rhs, Node):
            self.attrs._rhs._update_diff(context, -dy * self.attrs._log_lhs, **kwargs)

        if isinstance(self.attrs._lhs, Node):
            self.attrs._lhs._update_diff(
                context, -dy * get_gpu(self.attrs._rhs) / get_gpu(self.attrs._lhs), **kwargs)


class CrossEntropy:

    """This function evaluates the cross entropy loss
    between the target ``y`` and the input ``x``.

    .. math::
        E(x) = \sum_{n}^{N}\sum_{k}^{K}(-y*ln(x+\epsilon))

    :math:`N` is batch size.
    :math:`\epsilon` is small number for avoiding division by zero.

    Args:
        x (ndarray,Node): Input array.
        y (ndarray,Node): Target array.
        reduce_sum (bool): If True is given, the result array will be summed up and returns scalar value.

    Returns:
        (Node, ndarray): Cross entropy error.

    Raises:
        AssertionError: An assertion error will be raised if the given tensor dimension is less than 2.

    Example:
        >>> import renom as rm
        >>> import numpy as np
        >>>
        >>> x = np.array([[1.0, 0.5]])
        >>> y = np.array([[0.0, 1.0]])
        >>> print(x.shape, y.shape)
        ((1, 2), (1, 2))
        >>> loss = rm.cross_entropy(x, y)
        >>> print(loss)
        [0.6931471824645996]
        >>> loss = rm.cross_entropy(x, y, reduce_sum=False)
        >>> print(loss)
        [[0.          0.69314718]]

    """

    def __call__(self, lhs, rhs, reduce_sum=True):
        return cross_entropy(lhs, rhs, reduce_sum=reduce_sum)
