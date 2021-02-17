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
class hard_sigmoid(UnaryOp):

    @classmethod
    def _oper_cpu(cls, arg):
        return np.where(
            arg < -2.5, 0, np.where(2.5 <= arg, 1, 0.2 * arg + 0.5))

    @classmethod
    def _oper_gpu(cls, arg):
        ret = get_gpu(arg).empty_like_me()
        cu.cuhard_sigmoid_forward(get_gpu(arg), ret)
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            # self.attrs._arg._update_diff(
            #   context, self * (1. - self) * dy, **kwargs)
            self.attrs._arg._update_diff(
                context, np.where(self == 0.0, 0.0, np.where(
                    self == 1.0, 0.0, 0.2 * dy)), **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            dx = get_gpu(self.attrs._arg).empty_like_me()
            cu.cuhard_sigmoid_backward(get_gpu(self.attrs._arg), dx)
            self.attrs._arg._update_diff(context, dx * get_gpu(dy), **kwargs)


class HardSigmoid:
    '''HardSigmoid activation function as described by the following formula.

        ..math::
            ¥¥[f(x) ¥¥begin{cases}
            = 0 & (x ¥¥lt -2.5)
            = 0.2 * x + 0.5 & (-2.5 ¥¥leq x ¥¥lt 2.5)
            = 1 & (x ¥¥lt 2.5)
            ¥¥end{cases}¥¥]

    Args:
        x (ndarray, Node): Input numpy array or Node instance.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> x = np.array([1., -1.])
        >>> rm.hard_sigmoid(x)
        hard_sigmoid([ 0.7 ,  0.3])

        >>> # instantiation
        >>> activation = rm.HardSigmoid()
        >>> activation(x)
        hard_sigmoid([ 0.7 ,  0.3])

    '''

    def __call__(self, x):
        return hard_sigmoid(x)
