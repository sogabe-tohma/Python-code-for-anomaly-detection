#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from renom.core import UnaryOp, Node
from renom.debug_graph import showmark
import renom.cuda as cu
if cu.has_cuda():
    from renom.cuda.gpuvalue import get_gpu


@showmark
class softmax(UnaryOp):

    @classmethod
    def _oper_cpu(cls, arg):
        maxes = np.max(arg, axis=1, keepdims=True)
        u = np.exp(arg - maxes)
        summed = np.sum(u, axis=1, keepdims=True)
        z = u / (summed + 1e-8)
        return z

    @classmethod
    def _oper_gpu(cls, arg):
        z = get_gpu(arg).empty_like_me()
        with cu.cudnn_handler() as handle:
            cu.cuSoftmaxForward(handle, get_gpu(arg), z, mode=1)
        return z

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            dx = self * dy
            summed = dx - np.sum(dx, axis=1, keepdims=True)
            self.attrs._arg._update_diff(context, ((1.0 - self) * dy + summed) * self, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            with cu.cudnn_handler() as handle:
                dx = get_gpu(self).empty_like_me()
                cu.cuSoftmaxBackward(handle, get_gpu(self), get_gpu(dy), dx, mode=1)
            self.attrs._arg._update_diff(context, dx, **kwargs)


class Softmax:
    '''Soft max activation function
    is described by the following formula:

        :math:`f(x_j)=\\frac{exp(x_j)}{\sum_{i}exp(x_i)}`

    Args:
        x (ndarray, Variable): Input numpy array or instance of Variable.

    Example:
        >>> import renom as rm
        >>> import numpy as np
        >>> x = np.random.rand(1, 3)
        array([[ 0.11871966  0.48498547  0.7406374 ]])
        >>> z = rm.softmax(x)
        softmax([[ 0.23229694  0.33505085  0.43265226]])
        >>> np.sum(z, axis=1)
        array([ 1.])

        >>> # instantiation
        >>> activation = rm.Softmax()
        >>> activation(x)
        softmax([[ 0.23229694  0.33505085  0.43265226]])

    '''

    def __call__(self, x):
        return softmax(x)
