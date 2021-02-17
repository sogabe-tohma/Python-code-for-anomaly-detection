#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from renom.core import UnaryOp, Node
from renom.debug_graph import showmark
import renom.cuda as cu
if cu.has_cuda():
    from renom.cuda.gpuvalue import get_gpu


@showmark
class tanh(UnaryOp):

    @classmethod
    def _oper_cpu(cls, arg):
        return np.tanh(arg)

    @classmethod
    def _oper_gpu(cls, arg):
        ret = get_gpu(arg).empty_like_me()
        cu.cutanh(get_gpu(arg), ret)
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            self.attrs._arg._update_diff(context, (1.0 - self**2) * dy, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            self.attrs._arg._update_diff(context, (1.0 - self**2) * get_gpu(dy), **kwargs)


class Tanh:
    '''Hyperbolic tangent activation function as described by the following formula.

        :math:`f(x) = tanh(x)`

    Args:
        x (ndarray, Node): Input numpy array or Node instance.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> x = np.array([1., -1.])
        >>> rm.tanh(x)
        tanh([ 0.76159418, -0.76159418])

        >>> # instantiation
        >>> activation = rm.Tanh()
        >>> activation(x)
        tanh([ 0.76159418, -0.76159418])

    '''

    def __call__(self, x):
        return tanh(x)
