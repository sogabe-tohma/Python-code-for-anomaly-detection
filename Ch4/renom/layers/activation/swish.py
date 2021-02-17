#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from renom.core import UnaryOp, Node
from renom.debug_graph import showmark
import renom.cuda as cu
if cu.has_cuda():
    from renom.cuda.gpuvalue import get_gpu


@showmark
class swish(UnaryOp):

    def __new__(cls, arg, beta=1.0):
        return cls.calc_value(arg, beta)

    @classmethod
    def _oper_cpu(cls, arg, beta=1.0):
        ret = cls._create_node(arg / (1. + np.exp(-arg * beta)))
        ret.attrs._arg = arg
        ret.attrs._beta = beta
        return ret

    @classmethod
    def _oper_gpu(cls, arg, beta):
        z = get_gpu(arg).empty_like_me()
        cu.cuswish_forward(beta, get_gpu(arg), z)
        ret = cls._create_node(z)
        ret.attrs._arg = arg
        ret.attrs._beta = beta
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            beta = self.attrs._beta
            arg = self.attrs._arg
            self.attrs._arg._update_diff(
                context, (beta * self + (1. / (1. + np.exp(-arg * beta))) * (1. - beta * self)) * dy, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            beta = self.attrs._beta
            dx = get_gpu(self.attrs._arg).empty_like_me()
            cu.cuswish_backward(beta, get_gpu(self.attrs._arg), dx)
            self.attrs._arg._update_diff(context, dx * get_gpu(dy), **kwargs)


class Swish:
    '''Swish activation function as described by the following formula.

        :math:`f(beta, x) = x/(1 + \exp(-beta*x))`

    Args:
        x (ndarray, Node): Input numpy array or Node instance.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> x = np.array([1., -1.])
        >>> rm.swish(x)
        swish([ 0.7310586 , -0.26894143], dtype=float32)

        >>> # instantiation
        >>> activation = rm.Swish()
        >>> activation(x)
        swish([ 0.7310586 , -0.26894143], dtype=float32)

    '''

    def __init__(self, beta=1.0):
        self._beta = beta

    def __call__(self, x):
        return swish(x, self._beta)
