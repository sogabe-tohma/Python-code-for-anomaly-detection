#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from renom.core import UnaryOp, Node
from renom.debug_graph import showmark
import renom.cuda as cu
if cu.has_cuda():
    from renom.cuda.gpuvalue import get_gpu


@showmark
class relu6(UnaryOp):

    @classmethod
    def _oper_cpu(cls, arg):
        return np.clip(arg, 0, 6)

    @classmethod
    def _oper_gpu(cls, arg):
        ret = get_gpu(arg).empty_like_me()
        cu.curelu6_foward(get_gpu(arg), ret)
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            self.attrs._arg._update_diff(context, np.where(
                (0.0 < self) & (self < 6.0), dy, 0.0), **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            dx = get_gpu(self.attrs._arg).empty_like_me()
            cu.curelu6_backard(get_gpu(self.attrs._arg), dx)
            self.attrs._arg._update_diff(context, dx * get_gpu(dy), **kwargs)


class Relu6:
    '''Rectified Linear Unit (6) activation function as described by the following formula.

        :math:`f(x)=min(6,max(x, 0))`

    Args:
        x (ndarray, Node): Input numpy array or Node instance.

    Example:
        >>> import renom as rm
        >>> import numpy as np
        >>> x = np.array([[7, 1, -1]])
        array([[7, 1, -1]])
        >>> rm.relu6(x)
        relu([[0.  ,1.  , 0.]])

        >>> # instantiation
        >>> activation = rm.Relu6()
        >>> activation(x)
        relu([[0.  ,1.  , 0.]])

    '''

    def __call__(self, x):
        return relu6(x)
