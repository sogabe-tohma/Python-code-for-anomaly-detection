#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function
import numpy as np
from renom.core import Node, Variable, to_value
from renom.layers.function import Parametrized
from renom.utility.initializer import GlorotNormal
import renom as rm
from renom import precision
import renom.cuda as cu
if cu.has_cuda():
    from renom.cuda.gpuvalue import GPUValue, get_gpu

class group_normalize(Node):
    def __new__(cls, x, w, b, g, eps):
        return cls.calc_value(x, w, b, g, eps)

    @classmethod
    def _oper_cpu(cls, x, w, b, G, eps):
        assert len(x.shape) == 4, \
            "Input Data for Group Normalize layer must have 4 dimensions. Provided {}.".format(len(x.shape))
        N, C, H, W = x.shape
        G = min(G, C)
        x_group = np.reshape(to_value(x), (N, G, C // G, H, W))
        mean = np.mean(x_group, axis=(2, 3, 4), keepdims=True)
        var = np.var(x_group, axis=(2, 3, 4), keepdims=True)
        x_groupnorm = (x_group - mean) / np.sqrt(var + eps)
        x_norm = np.reshape(x_groupnorm, (N, C, H, W))

        out = x_norm * to_value(w) + to_value(b)

        ret = cls._create_node(out)
        ret.attrs._x = x
        ret.attrs._x_norm = x_norm
        ret.attrs._g = G
        ret.attrs._m = mean
        ret.attrs._v = var
        ret.attrs._w = w
        ret.attrs._b = b
        ret.attrs._eps = eps

        return ret

    @classmethod
    def _oper_gpu(cls, x, w, b, G, eps):
        assert len(x.shape) == 4, "Group Normalization supports only CNN model for now. Data should have 4 dimensions."
        N, C, H, W = x.shape
        G = min(G, C)
        x_group = rm.reshape(x, (N, G, C // G, H, W))
        mean = rm.mean(x_group, axis=(2, 3, 4), keepdims=True)
        size = x_group.shape[2] * x_group.shape[3] * x_group.shape[4]
        var = rm.sum(rm.square(x_group - mean), axis=(2, 3, 4), keepdims=True) / size

        x_groupnorm = (x_group - mean) / rm.sqrt(var + eps)

        x_norm = rm.reshape(x_groupnorm, (N, C, H, W))

        out = x_norm * w + b

        ret = cls._create_node(get_gpu(out))
        ret.attrs._x = x
        ret.attrs._x_norm = x_norm
        ret.attrs._g = G
        ret.attrs._m = mean
        ret.attrs._v = var
        ret.attrs._w = w
        ret.attrs._b = b
        ret.attrs._eps = eps

        return ret


    def _backward_cpu(self, context, dy, **kwargs):
        N, C, H, W = dy.shape
        G, x, x_norm, mean = self.attrs._g, self.attrs._x, self.attrs._x_norm, self.attrs._m
        var, w, eps = self.attrs._v, self.attrs._w, self.attrs._eps

        if isinstance(self.attrs._w, Node):
            self.attrs._w._update_diff(context, np.sum(dy * to_value(x_norm), axis=(0, 2, 3), keepdims=True), **kwargs)
        if isinstance(self.attrs._b, Node):
            self.attrs._b._update_diff(context, np.sum(dy, axis=(0, 2, 3), keepdims=True), **kwargs)

        if isinstance(self.attrs._x, Node):
            # dx_group，(N, G, C // G, H, W)
            # dx_groupnorm
            dx_norm = dy * to_value(w)

            dx_groupnorm = dx_norm.reshape((N, G, C // G, H, W))

            # dvar
            x_group = x.reshape((N, G, C // G, H, W))

            dvar = np.sum(dx_groupnorm * -1.0 / 2 * (x_group - mean) /
                          (var + eps) ** (3.0 / 2), axis=(2, 3, 4), keepdims=True)

            # dmean
            N_GROUP = (C // G) * H * W
            dmean1 = np.sum(dx_groupnorm * -1.0 / np.sqrt(var + eps), axis=(2, 3, 4), keepdims=True)

            dmean2_var = dvar * -2.0 / N_GROUP * np.sum(x_group - mean, axis=(2, 3, 4), keepdims=True)
            dmean = dmean1 + dmean2_var

            # dx_group
            dx_group1 = dx_groupnorm * 1.0 / np.sqrt(var + eps)
            dx_group2_mean = dmean * 1.0 / N_GROUP
            dx_group3_var = dvar * 2.0 / N_GROUP * (x_group - mean)
            dx_group = dx_group1 + dx_group2_mean + dx_group3_var

            # dx
            dx = dx_group.reshape((N, C, H, W))

            self.attrs._x._update_diff(context, dx, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        N, C, H, W = dy.shape
        G, x, x_norm, mean, var, w, b, eps = map(get_gpu, (self.attrs._g, self.attrs._x, self.attrs._x_norm,
                                                 self.attrs._m, self.attrs._v, self.attrs._w, self.attrs._b,
                                                 self.attrs._eps))

        if isinstance(self.attrs._w, Node):
            self.attrs._w._update_diff(context, rm.sum(dy * x_norm, axis=(0, 2, 3), keepdims=True), **kwargs)
        if isinstance(self.attrs._b, Node):
            self.attrs._b._update_diff(context, rm.sum(dy, axis=(0, 2, 3), keepdims=True), **kwargs)

        if isinstance(self.attrs._x, Node):
            # dx_group，(N, G, C // G, H, W)
            # dx_groupnorm
            dx_norm = dy * w

            dx_groupnorm = dx_norm.reshape((N, G, C // G, H, W))

            # dvar
            x_group = x.reshape((N, G, C // G, H, W))

            dvar = rm.sum(dx_groupnorm * -1.0 / 2 * (x_group - mean) /
                          (var + eps) ** (3.0 / 2), axis=(2, 3, 4), keepdims=True)

            # dmean
            N_GROUP = (C // G) * H * W
            dmean1 = rm.sum(dx_groupnorm * -1.0 / rm.sqrt(var + eps), axis=(2, 3, 4), keepdims=True)

            dmean2_var = dvar * -2.0 / N_GROUP * rm.sum(x_group - mean, axis=(2, 3, 4), keepdims=True)
            dmean = dmean1 + dmean2_var

            # dx_group
            dx_group1 = dx_groupnorm * 1.0 / rm.sqrt(var + eps)
            dx_group2_mean = dmean * 1.0 / N_GROUP
            dx_group3_var = dvar * 2.0 / N_GROUP * (x_group - mean)
            dx_group = dx_group1 + dx_group2_mean + dx_group3_var

            # dx
            dx = dx_group.reshape((N, C, H, W))

            self.attrs._x._update_diff(context, dx, **kwargs)



class GroupNormalize(Parametrized):
    SERIALIZED = ('_epsilon', '_group')

    def __init__(self, input_size=None, epsilon=1e-5, initializer=None, group=32, weight_decay=0):
        self._epsilon = epsilon
        self._weight_decay = weight_decay
        self._group = group
        self._initializer = initializer
        super(GroupNormalize, self).__init__(input_size)

    def weight_initiallize(self, input_size):
        size_i = [1, ]
        size_i.extend(input_size)
        if len(size_i) > 2:
            size_i[2] = 1
            size_i[3] = 1
        if self._initializer is None:
            weights = Variable(np.ones(size_i, dtype=precision), auto_update=True, weight_decay=self._weight_decay)
        else:
            weights = Variable(self._initializer(size_i).astype(precision),
                               auto_update=True, weight_decay=self._weight_decay)
        self.params = {"w": weights}
        self.params["b"] = Variable(np.zeros(size_i, dtype=precision), auto_update=True)

    def forward(self, x):
        ret = group_normalize(x, self.params['w'], self.params['b'], self._group, self._epsilon)
        return ret
