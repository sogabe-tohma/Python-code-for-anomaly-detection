#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from renom.core import Node
from renom import precision
from renom.layers.function.parameterized import Model
import renom.cuda as cu
if cu.has_cuda():
    from renom.cuda.gpuvalue import get_gpu, GPUValue

try:
    from renom.cuda import curand_generator
except ImportError:
    pass


class dropout(Node):

    def __new__(cls, x, dropout_ratio=0.5, inference=False):
        if inference:
            return x
        ret = cls.calc_value(x, 1. - dropout_ratio)
        ret._ratio = dropout_ratio
        return ret

    @classmethod
    def _oper_cpu(cls, x, dropout_ratio):
        mask = np.array(np.random.rand(*x.shape) < dropout_ratio, dtype=precision) / dropout_ratio
        value = x * mask

        ret = cls._create_node(value)
        ret.attrs._x = x
        ret.attrs._mask = mask
        return ret

    @classmethod
    def _oper_gpu(cls, x, dropout_ratio):
        mask = get_gpu(x).empty_like_me()
        curand_generator().rand_bernoulli(mask, 1 - dropout_ratio)
        mask = mask / dropout_ratio
        value = get_gpu(x) * mask
        ret = cls._create_node(value)
        ret.attrs._x = x
        ret.attrs._mask = mask
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._x, Node):
            dx = self.attrs._mask * dy
            self.attrs._x._update_diff(context, dx, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._x, Node):
            dx = get_gpu(self.attrs._mask) * get_gpu(dy)
            self.attrs._x._update_diff(context, dx, **kwargs)


class spatial_dropout(dropout):

    def __new__(cls, x, dropout_ratio=0.5, inference=False):
        assert len(x.shape) == 4, "Spatial_dropout only accepts 4d tensors."
        if inference:
            return x
        else:
            return cls.calc_value(x, 1. - dropout_ratio)

    @classmethod
    def _oper_cpu(cls, x, dropout_ratio):
        mask = np.array(np.random.rand(*x.shape[:2]) < dropout_ratio,
                        dtype=precision)[:, :, None, None] / dropout_ratio
        value = x * mask
        ret = cls._create_node(value)
        ret.attrs._x = x
        ret.attrs._mask = mask
        return ret

    @classmethod
    def _oper_gpu(cls, x, drop_out_ratio):
        shape = (x.shape[0], x.shape[1], 1, 1)
        mask = GPUValue(shape=shape)
        curand_generator().rand_bernoulli(mask, 1 - drop_out_ratio)
        mask = mask / drop_out_ratio
        mask = mask * get_gpu(x).ones_like_me()
        value = get_gpu(x) * get_gpu(mask)
        ret = cls._create_node(value)
        ret.attrs._x = x
        ret.attrs._mask = mask
        return ret


class Dropout(Model):
    """Applies Dropout [dropout]_ to the input.

    Dropout function randomly selects a fraction (specified by dropout_ratio) of
    the data sets them to zero.
    Remaining data will be rescaled by ``1/(1 - dropout_ratio)``.

    Args:
        dropout_ratio (float): Dropout ratio.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>>
        >>> x = rm.Variable(np.random.rand(3, 2))
        >>> x
        Variable([[ 0.49312586,  0.95414829],
                  [ 0.7346437 ,  0.9014256 ],
                  [ 0.09413767,  0.29910043]], dtype=float32)
        >>> layer = rm.Dropout(0.2)
        >>> z = layer(x)
        >>> z
        dropout([[ 0.        ,  1.19268537],
                 [ 0.91830462,  1.12678194],
                 [ 0.        ,  0.37387553]], dtype=float32)

        >>> z = rm.dropout(x, 0.75)
        >>> z
        dropout([[ 0.65750116,  0.        ],
                 [ 0.        ,  1.20190084],
                 [ 0.12551689,  0.39880058]], dtype=float32)

    .. [dropout] Hinton, Geoffrey E.; Srivastava, Nitish; Krizhevsky, Alex; Sutskever,
                                                Ilya; Salakhutdinov, Ruslan R. (2012).
        Improving neural networks by preventing co-adaptation of feature detectors

    """

    def __init__(self, dropout_ratio=0.5):
        self._dropout_ratio = dropout_ratio
        self.inference = False

    def __call__(self, x):
        if self.inference:
            return x
        return self.forward(x)

    def forward(self, x):
        return dropout(x, self._dropout_ratio, self.inference)


class SpatialDropout(Dropout):
    """Applies spatial dropout to the input.
    This function drops feature maps randomly.

    Args:
        dropout_ratio (float): Dropout ratio.

    Raises:
        AssertionError: An assertion error will be raised if the input tensor dimension is not 4.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> x = rm.Variable(np.random.rand(3, 3, 1, 1))
        Variable([[[[ 0.55784005]],
                   [[ 0.99528867]],
                   [[ 0.77544725]]],
                  [[[ 0.65406305]],
                   [[ 0.27961349]],
                   [[ 0.43104461]]],
                  [[[ 0.66176379]],
                   [[ 0.70499772]],
                   [[ 0.87102354]]]], dtype=float32)
        >>> z = rm.spatial_dropout(x)
        >>> z
        spatial_dropout([[[[ 1.1156801 ]],
                          [[ 0.        ]],
                          [[ 1.5508945 ]]],
                         [[[ 1.30812609]],
                          [[ 0.55922699]],
                          [[ 0.        ]]],
                         [[[ 0.        ]],
                          [[ 1.40999544]],
                          [[ 1.74204707]]]], dtype=float32)

    Note:
        SpatialDropout only accepts 4d tensor data.

    """

    def forward(self, x):
        return spatial_dropout(x, self._dropout_ratio, self.inference)
