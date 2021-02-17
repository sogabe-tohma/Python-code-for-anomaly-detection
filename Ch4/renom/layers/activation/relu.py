#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from renom.core import UnaryOp, Node
from renom.debug_graph import showmark
import renom.cuda as cu
if cu.has_cuda():
    from renom.cuda.gpuvalue import get_gpu


@showmark
class relu(UnaryOp):

    @classmethod
    def _oper_cpu(cls, arg):
        return np.maximum(arg, 0)

    @classmethod
    def _oper_gpu(cls, arg):
        ret = get_gpu(arg).empty_like_me()
        cu.curelu_foward(get_gpu(arg), ret)
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            self.attrs._arg._update_diff(context, np.where(self == 0, 0, dy), **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            dx = get_gpu(self.attrs._arg).empty_like_me()
            cu.curelu_backard(get_gpu(self.attrs._arg), dx)
            self.attrs._arg._update_diff(context, dx * get_gpu(dy), **kwargs)


class Relu:
    '''Rectified Linear Unit activation function as described by the following formula.

        :math:`f(x)=max(x, 0)`

    Args:
        x (ndarray, Node): Input numpy array or Node instance.

    Example:
        >>> import renom as rm
        >>> import numpy as np
        >>> x = np.array([[1, -1]])
        array([[ 1, -1]])
        >>> rm.relu(x)
        relu([[ 1.  , 0.]])

        >>> # instantiation
        >>> activation = rm.Relu()
        >>> activation(x)
        relu([[ 1.  , 0]])

    '''

    def __call__(self, x):
        return relu(x)
