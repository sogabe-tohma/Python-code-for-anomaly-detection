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
class sigmoid(UnaryOp):

    @classmethod
    def _oper_cpu(cls, arg):
        return 1. / (1. + np.exp(-arg))

    @classmethod
    def _oper_gpu(cls, arg):
        ret = get_gpu(arg).empty_like_me()
        cu.cusigmoid(get_gpu(arg), ret)
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            self.attrs._arg._update_diff(context, self * (1. - self) * dy, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            s = get_gpu(self)
            self.attrs._arg._update_diff(context, s * (-s + 1.) * get_gpu(dy), **kwargs)


class Sigmoid:
    '''Sigmoid activation function as described by the following formula.

        :math:`f(x) = 1/(1 + \exp(-x))`

    Args:
        x (ndarray, Node): Input numpy array or Node instance.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> x = np.array([1., -1.])
        >>> rm.sigmoid(x)
        sigmoid([ 0.7310586 ,  0.26894143])

        >>> # instantiation
        >>> activation = rm.Sigmoid()
        >>> activation(x)
        sigmoid([ 0.7310586 ,  0.26894143])

    '''

    def __call__(self, x):
        return sigmoid(x)
