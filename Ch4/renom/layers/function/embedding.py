#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from renom.core import Node, Variable
from renom import precision
from renom.layers.function.parameterized import Parametrized
from renom.utility.initializer import GlorotNormal
import renom.cuda as cu
if cu.has_cuda():
    from renom.cuda.gpuvalue import GPUValue, get_gpu


class embedding(Node):

    def __new__(cls, x, w):
        assert x.shape[1] == 1
        return cls.calc_value(x, w)

    @classmethod
    def _oper_cpu(cls, x, w):
        if isinstance(x, Node):
            index = x.as_ndarray().astype(np.int)[:, 0]
        else:
            index = x.astype(np.int)[:, 0]
        value = w[index]
        ret = cls._create_node(value)
        ret.attrs._x = x
        ret.attrs._w = w
        ret.attrs._index = index
        return ret

    @classmethod
    def _oper_gpu(cls, x, w):
        z = GPUValue(shape=(len(x), len(w[0])))
        cu.cuembedding_forward(get_gpu(x), get_gpu(w), z)
        ret = cls._create_node(z)
        ret.attrs._x = x
        ret.attrs._w = w
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._w, Node):
            N = len(self.attrs._index)
            dx = np.zeros(self.attrs._w.shape, dtype=self.attrs._w.dtype)
            for i in range(N):
                dx[self.attrs._index[i]] += dy[i]
            self.attrs._w._update_diff(context, dx, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._w, Node):
            dx = get_gpu(self.attrs._w).zeros_like_me()
            cu.cuembedding_backward(get_gpu(self.attrs._x), get_gpu(dy), dx)
            self.attrs._w._update_diff(context, dx, **kwargs)


class Embedding(Parametrized):
    """Embedding layer.
    This layer is the special case of dense layer. The case is that the input value is onehot encoded.
    Since the onehot encoded input is very sparse, the matrix product performed in the dense layer is redundant.

    The difference between dense layer and embedding layer is bellow.

    | **[Dense layer]**
    |  data -> onehot encoding -> onehot data -> dense layer -> embedded data

    | **[Embedding layer]**
    |  data -> embedding layer -> embedded data

    Args:
        output_size (int): Output unit size.
        input_size (int): Input unit size. This is same as number of embedding characters.
        initializer (Initializer): Initializer object for weight initialization.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> N = 4
        >>> a = np.arange(N).reshape(N, 1)
        >>> layer = rm.Embedding(output_size=1, input_size=8)
        >>> out = layer(a)

    Note:
        1. This layer only accepts matrix which shape is (N, 1) and has integer value. *N is batch size.
        2. Both ``output_size`` and ``input_size`` must be specified.
    """

    def __init__(self, output_size, input_size, initializer=GlorotNormal(), weight_decay=None):
        self._output_size = output_size
        self._initializer = initializer
        self._weight_decay = weight_decay
        super(Embedding, self).__init__(input_size)

    def weight_initiallize(self, input_size):
        size_i = input_size[0] if isinstance(input_size, tuple) else input_size
        size_o = self._output_size
        self.params = {
            "w": Variable(self._initializer((size_i, size_o)), auto_update=True, weight_decay=self._weight_decay)}

    def forward(self, x):
        return embedding(x, self.params.w)
