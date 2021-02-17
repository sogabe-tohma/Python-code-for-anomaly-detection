#!/usr/bin/env python
# coding: utf-8

# In[14]:


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
class hard_tanh(UnaryOp):
    @classmethod
    def _oper_cpu(cls, arg):
        return np.maximum(-1, np.minimum(1, arg))

    @classmethod
    def _oper_gpu(cls, arg):
        ret = get_gpu(arg).empty_like_me()
        cu.cuhard_tanh_forward(get_gpu(arg), ret)
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            self.attrs._arg._update_diff(context, np.where(
                self == -1, 0, np.where(self == 1, 0, dy)), **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            dx = get_gpu(self.attrs._arg).empty_like_me()
            cu.cuhard_tanh_backward(get_gpu(self.attrs._arg), dx)
            self.attrs._arg._update_diff(context, dx * get_gpu(dy), **kwargs)


class Hard_Tanh(object):
    '''Hard hyperbolic tangent activation function as described by the following formula.

        :math:`f(x) = max(-1, min(1, x))`

    Args:
        x (ndarray, Node): Input numpy array or Node instance.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> x = np.array([1., -1.])
        >>> rm.hard_tanh(x)
        hard_tanh([ 1., -1.])

        >>> # instantiation
        >>> activation = rm.Hard_Tanh()
        >>> activation(x)
        hard_tanh([ 1., -1.])

    '''

    def __call__(self, x):
        return hard_tanh(x)
