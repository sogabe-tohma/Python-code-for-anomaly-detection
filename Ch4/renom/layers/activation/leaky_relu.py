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
class leaky_relu(UnaryOp):

    def __new__(cls, arg, slope=0.01):
        return cls.calc_value(arg, slope)

    @classmethod
    def _oper_cpu(cls, arg, slope):
        ret = cls._create_node(np.where(arg > 0, arg, arg * slope))
        ret.attrs._arg = arg
        ret.attrs._slope = slope
        return ret

    @classmethod
    def _oper_gpu(cls, arg, slope):
        z = get_gpu(arg).empty_like_me()
        cu.culeaky_leru_forward(slope, get_gpu(arg), z)
        ret = cls._create_node(z)
        ret.attrs._arg = arg
        ret.attrs._slope = slope
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            slope = self.attrs._slope
            self.attrs._arg._update_diff(context, np.where(self > 0, dy, dy * slope), **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            slope = self.attrs._slope
            dx = get_gpu(self.attrs._arg).empty_like_me()
            cu.culeaky_leru_backward(slope, get_gpu(self.attrs._arg), dx)
            self.attrs._arg._update_diff(context, dx * get_gpu(dy), **kwargs)


class LeakyRelu:

    '''The Leaky relu [leaky_relu]_ activation function is described by the following formula:

        :math:`f(x)=max(x, 0)+min(slope*x, 0)`

    Args:
        x (ndarray, Variable): Input numpy array or instance of Variable.
        slope (float): Coefficient multiplied by negative values.

    Example:
        >>> import renom as rm
        >>> import numpy as np
        >>> x = np.array([[1, -1]])
        array([[ 1, -1]])
        >>> rm.leaky_relu(x, slope=0.01)
        leaky_relu([[ 1.  , -0.01]])

        >>> # instantiation
        >>> activation = rm.LeakyRelu(slope=0.01)
        >>> activation(x)
        leaky_relu([[ 1.  , -0.01]])

    .. [leaky_relu] Andrew L. Maas, Awni Y. Hannun, Andrew Y. Ng (2014).
        Rectifier Nonlinearities Improve Neural Network Acoustic Models
    '''

    def __init__(self, slope=0.01):
        self._slope = slope

    def __call__(self, x):
        return leaky_relu(x, self._slope)
