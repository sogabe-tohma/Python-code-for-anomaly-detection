#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from renom.core import UnaryOp, Node
from renom.debug_graph import showmark
from renom.operation import where
from renom.config import precision
import renom.cuda as cu
if cu.has_cuda():
    from renom.cuda.gpuvalue import get_gpu


@showmark
class elu(UnaryOp):

    def __new__(cls, arg, alpha=0.01):
        return cls.calc_value(arg, alpha)

    @classmethod
    def _oper_cpu(cls, arg, alpha):
        ret = cls._create_node(np.where(arg > 0, arg, (np.exp(arg) - 1) * alpha))
        ret.attrs._arg = arg
        ret.attrs._alpha = alpha
        return ret

    @classmethod
    def _oper_gpu(cls, arg, alpha):
        z = get_gpu(arg).empty_like_me()
        cu.cueru_forward(alpha, get_gpu(arg), z)
        ret = cls._create_node(z)
        ret.attrs._arg = arg
        ret.attrs._alpha = alpha
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            alpha = self.attrs._alpha
            self.attrs._arg._update_diff(context, np.where(
                self > 0, dy, (alpha + self) * dy), **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            alpha = self.attrs._alpha
            dx = get_gpu(self.attrs._arg).empty_like_me()
            cu.cueru_backward(alpha, get_gpu(self.attrs._arg), dx)
            self.attrs._arg._update_diff(context, dx * get_gpu(dy), **kwargs)


class Elu:

    '''The Exponential Linear Units [elu]_ activation function
    is described by the following formula:

        :math:`f(x)=max(x, 0) + alpha*min(exp(x)-1, 0)`

    Args:
        x (ndarray, Variable): Input numpy array or instance of Variable.
        alpha (float): Coefficient multiplied by exponentiated values.

    Example:
        >>> import renom as rm
        >>> import numpy as np
        >>> x = np.array([[1, -1]])
        array([[ 1, -1]])
        >>> rm.elu(x)
        elu([[ 1.  , -0.00632121]])

        >>> # instantiation
        >>> activation = rm.Elu()
        >>> activation(x)
        elu([[ 1.  , -0.00632121]])

    .. [elu] Djork-Arné Clevert, Thomas Unterthiner, Sepp Hochreiter (2015).
        Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs).
        Published as a conference paper at ICLR 2016
    '''

    def __init__(self, alpha=0.01):
        self._alpha = alpha

    def __call__(self, x):
        return elu(x, self._alpha)
