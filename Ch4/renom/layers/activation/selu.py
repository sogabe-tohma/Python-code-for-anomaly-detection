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
class selu(UnaryOp):

    def __new__(cls, arg):
        alpha = 1.6732632423543772848170429916717
        lmda = 1.0507009873554804934193349852946
        return cls.calc_value(arg, alpha, lmda)

    @classmethod
    def _oper_cpu(cls, arg, alpha, lmda):
        ret = cls._create_node(np.where(arg > 0, arg, (np.exp(arg) - 1) * alpha) * lmda)
        ret.attrs._arg = arg
        ret.attrs._alpha = alpha
        ret.attrs._lmda = lmda
        return ret

    @classmethod
    def _oper_gpu(cls, arg, alpha, lmda):
        z = get_gpu(arg).empty_like_me()
        cu.cueru_forward(alpha, get_gpu(arg), z)
        ret = cls._create_node(z * lmda)
        ret.attrs._arg = arg
        ret.attrs._alpha = alpha
        ret.attrs._lmda = lmda
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            alpha = self.attrs._alpha
            lmda = self.attrs._lmda
            self.attrs._arg._update_diff(context, np.where(
                self > 0, dy, (alpha + self) * dy) * lmda, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            alpha = self.attrs._alpha
            lmda = self.attrs._lmda
            dx = get_gpu(self.attrs._arg).empty_like_me()
            cu.cueru_backward(alpha, get_gpu(self.attrs._arg), dx)
            self.attrs._arg._update_diff(context, dx * get_gpu(dy) * lmda, **kwargs)


class Selu:

    '''The scaled exponential linear unit [selu]_ activation
    function is described by the following formula:

        :math:`a = 1.6732632423543772848170429916717`
        :math:`b = 1.0507009873554804934193349852946`
        :math:`f(x) = b*max(x, 0)+min(0, exp(x) - a)`

    Args:
        x (ndarray, Node): Input numpy array or Node instance.

    Example:
        >>> import renom as rm
        >>> import numpy as np
        >>> x = np.array([[1, -1]])
        array([[ 1, -1]])
        >>> rm.relu(x)
        selu([ 1.05070102, -1.11133075])

        >>> # instantiation
        >>> activation = rm.Relu()
        >>> activation(x)
        selu([ 1.05070102, -1.11133075])

    .. [selu] Günter Klambauer, Thomas Unterthiner, Andreas Mayr, Sepp Hochreiter.
        Self-Normalizing Neural Networks.
        Learning (cs.LG); Machine Learning
    '''

    def __call__(self, x):
        return selu(x)
