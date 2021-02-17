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
class mish(UnaryOp):
    MISH_THRESHOLD = 20.

    @classmethod
    def _oper_cpu(cls, arg):
        x = np.clip(arg, None, cls.MISH_THRESHOLD)
        return arg * np.tanh(np.log(1. + np.exp(x)))

    @classmethod
    def _oper_gpu(cls, arg):
        ret = get_gpu(arg).empty_like_me()
        cu.cumish_forward(get_gpu(arg), ret)
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            arg = self.attrs._arg
            sp = np.clip(arg, None, self.MISH_THRESHOLD)
            sp = np.log(np.exp(sp) + 1.)
            grad_sp = 1. - np.exp(-sp)
            tsp = np.tanh(sp)
            grad_tsp = (1 - tsp * tsp) * grad_sp
            self.attrs._arg._update_diff(
                context, (arg * grad_tsp + tsp) * dy, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            dx = get_gpu(self.attrs._arg).empty_like_me()
            cu.cumish_backward(get_gpu(self.attrs._arg), dx)
            self.attrs._arg._update_diff(context, dx * get_gpu(dy), **kwargs)


class Mish:
    '''Mish activation function as described by the following formula.

        ..math::
            mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))

    Args:
        x (ndarray, Node): Input numpy array or Node instance.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> x = np.array([1., -1.])
        >>> rm.mish(x)
        mish([ 0.86509842, -0.30340147], dtype=float32)

        >>> # instantiation
        >>> activation = rm.Mish()
        >>> activation(x)
        mish([ 0.86509842, -0.30340147], dtype=float32)
    '''

    def __call__(self, x):
        return mish(x)
