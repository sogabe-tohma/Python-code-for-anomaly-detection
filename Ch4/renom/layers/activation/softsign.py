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
class softsign(UnaryOp):

    @classmethod
    def _oper_cpu(cls, arg):
        return arg / (1. + np.abs(arg))

    @classmethod
    def _oper_gpu(cls, arg):
        ret = get_gpu(arg).empty_like_me()
        cu.cusoftsign_forward(get_gpu(arg), ret)
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            dx = 1. / np.square(1. + np.abs(self.attrs._arg))
            self.attrs._arg._update_diff(context, dx * dy, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            dx = get_gpu(self.attrs._arg).empty_like_me()
            cu.cusoftsign_backward(get_gpu(self.attrs._arg), dx)
            self.attrs._arg._update_diff(context, dx * get_gpu(dy), **kwargs)


class Softsign:
    '''Softsign activation function as described by the following formula.

        :math:`f(x) = x / (1 + |x|)`

    Args:
        x (ndarray, Node): Input numpy array or Node instance.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> x = np.array([1., -1.])
        >>> rm.softsign(x)
        fill in actual values here!!!!
        softsign([ 0.7310586 ,  0.26894143])

        >>> # instantiation
        >>> activation = rm.Softsign()
        >>> activation(x)
        fill in actual values here!!!
        softsign([ 0.7310586 ,  0.26894143])

    '''

    def __call__(self, x):
        return softsign(x)
