#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import numpy as np
import renom as rm
from renom.core import BinOp, Node, to_value
import renom.cuda as cu
if cu.has_cuda():
    from renom.cuda.gpuvalue import get_gpu
from renom.operation import where


class smoothed_l1(Node):
    def __new__(cls, lhs, rhs, delta=1.0, reduce_sum=True):
        assert rhs.ndim > 1, "Input arrays must have no less than 2 dimension."
        return cls.calc_value(lhs, rhs, delta, reduce_sum)

    @classmethod
    def _oper_cpu(cls, lhs, rhs, delta, reduce_sum):
        assert rhs.ndim > 1, "Input arrays must have no less than 2 dimension."
        N = float(lhs.shape[0])
        d = lhs - rhs
        abs_d = abs(d)
        if reduce_sum:
            loss = np.sum(np.where(abs_d < delta, 0.5 * d * d, delta * (abs_d - 0.5 * delta)))
        else:
            loss = np.where(abs_d < delta, 0.5 * d * d, delta * (abs_d - 0.5 * delta))
        ret = cls._create_node(loss / N)
        ret.attrs._delta = delta
        ret.attrs._lhs = lhs
        ret.attrs._rhs = rhs
        ret.attrs._d = d
        return ret

    @classmethod
    def _oper_gpu(cls, lhs, rhs, delta, reduce_sum):
        assert rhs.ndim > 1, "Input arrays must have no less than 2 dimension."
        N = float(lhs.shape[0])
        d = lhs - rhs
        abs_d = abs(d.as_ndarray())
        flag = abs_d < delta
        if reduce_sum:
            loss = cu.cusum(get_gpu(flag * 0.5 * (d * d) +
                                    (1 - flag) * (abs_d - 0.5 * delta) * delta))
        else:
            loss = get_gpu(flag * 0.5 * (d * d) + (1 - flag) * (abs_d - 0.5 * delta) * delta)
        ret = cls._create_node(loss / N)
        ret.attrs._delta = delta
        ret.attrs._lhs = lhs
        ret.attrs._rhs = rhs
        ret.attrs._d = d
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._lhs, Node):
            N = float(self.attrs._lhs.shape[0])
            delta = self.attrs._delta
            mask = abs(self.attrs._d) < delta
            dx = np.where(mask, self.attrs._d, np.sign(self.attrs._d) * delta)
            self.attrs._lhs._update_diff(context, dx * dy / N, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._lhs, Node):
            N = float(self.attrs._lhs.shape[0])
            delta = self.attrs._delta
            mask = abs(self.attrs._d) <= delta
            sign = (self.attrs._d > 0) * 2 - 1
            dx = mask * self.attrs._d + (1 - mask) * sign * delta
            self.attrs._lhs._update_diff(context, dx * dy / N, **kwargs)


class SmoothedL1(object):
    def __init__(self, delta, reduce_sum=True):
        self._delta = delta
        self._reduce_sum = reduce_sum

    def __call__(self, x, y):
        return smoothed_l1(x, y, self._delta, self._reduce_sum)
