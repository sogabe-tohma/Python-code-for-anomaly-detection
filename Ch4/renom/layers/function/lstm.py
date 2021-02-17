#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from renom.layers.activation.sigmoid import sigmoid
from renom.layers.activation.tanh import tanh
from renom.core import Node, Variable
from renom import precision
from renom.operation import dot, sum
from renom.utility.initializer import GlorotNormal
from .parameterized import Parametrized
import renom.cuda as cu
if cu.has_cuda():
    from renom.cuda.gpuvalue import GPUValue, get_gpu


def gate(x):
    return 1. / (1. + np.exp(-x))


def activation(x):
    return np.tanh(x)


def gate_diff(x):
    return x * (- x + 1.)


def activation_diff(x):
    return (1.0 - x**2)


class lstm(Node):
    def __new__(cls, x, pz, ps, w, wr, b):
        return cls.calc_value(x, pz, ps, w, wr, b)

    @classmethod
    def _oper_cpu(cls, x, pz, ps, w, wr, b):
        s = np.zeros((x.shape[0], w.shape[1] // 4), dtype=precision) if ps is None else ps
        z = np.zeros((x.shape[0], w.shape[1] // 4), dtype=precision) if pz is None else pz

        u = dot(x, w) + dot(z, wr)
        if b is not None:
            u += b
        m = u.shape[1] // 4
        u, gated = np.split(u.as_ndarray(), [m, ], axis=1)
        u = tanh(u)

        gated = sigmoid(gated)

        state = gated[:, m:m * 2] * u + gated[:, :m] * s
        z = tanh(state) * gated[:, m * 2:]

        ret = cls._create_node(z)
        ret.attrs._x = x
        ret.attrs._w = w
        ret.attrs._wr = wr
        ret.attrs._b = b
        ret.attrs._pz = pz
        ret.attrs._u = u
        ret.attrs._pstate = ps
        ret.attrs._state = state
        ret.attrs._gated = gated
        ret._state = state

        if isinstance(pz, Node):
            pz.attrs._pfgate = gated[:, :m]

        return ret

    @classmethod
    def _oper_gpu(cls, x, pz, ps, w, wr, b):
        if ps is None:
            tmp = GPUValue(shape=(x.shape[0], w.shape[1] // 4))
            s_p = tmp.zeros_like_me()
            z_p = tmp.zeros_like_me()
        else:
            s_p = ps
            z_p = get_gpu(pz)

        u = dot(x, w) + dot(z_p, wr)
        if b is not None:
            u += b

        z = get_gpu(z_p).empty_like_me()
        state = get_gpu(s_p).empty_like_me()

        cu.culstm_forward_activate(get_gpu(u))
        cu.culstm_forward(get_gpu(u), get_gpu(state), get_gpu(s_p), get_gpu(z))

        ret = cls._create_node(z)

        ret.attrs._x = x
        ret.attrs._w = w
        ret.attrs._wr = wr
        ret.attrs._b = b
        ret.attrs._pz = pz
        ret.attrs._u = u
        ret.attrs._pstate = s_p
        ret.attrs._state = state
        ret._state = state

        if isinstance(pz, Node):
            pz.attrs._pfgate = u

        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        n, m = dy.shape

        w = self.attrs._w
        wr = self.attrs._wr
        b = self.attrs._b

        u = self.attrs._u
        s = tanh(self.attrs._state)

        gated = self.attrs._gated
        gd = gate_diff(gated)
        ps = self.attrs._pstate

        drt = context.restore(wr, np.zeros((n, m * 4), dtype=dy.dtype))
        dou = context.restore(w, np.zeros((n, m), dtype=dy.dtype))

        pfg = self.attrs.get("_pfgate", np.zeros_like(self))

        e = dy

        do = e * s * gd[:, 2 * m:]
        dou = e * gated[:, 2 * m:] * activation_diff(s) + pfg * dou

        df = dou * gd[:, :m] * ps if ps is not None else np.zeros_like(dou)
        di = dou * gd[:, m:2 * m] * u
        dc = dou * activation_diff(u) * gated[:, m:2 * m]

        dr = np.hstack((dc, df, di, do))
        dx = np.dot(dr, w.T)

        context.store(wr, dr)
        context.store(w, dou)

        if isinstance(self.attrs._x, Node):
            self.attrs._x._update_diff(context, dx)

        if isinstance(w, Node):
            w._update_diff(context, np.dot(self.attrs._x.T, dr))

        if isinstance(wr, Node):
            wr._update_diff(context, np.dot(self.T, drt))

        if isinstance(b, Node):
            b._update_diff(context, np.sum(dr, axis=0, keepdims=True))

        if isinstance(self.attrs._pz, Node):
            self.attrs._pz._update_diff(context, np.dot(dr, wr.T))

    def _backward_gpu(self, context, dy, **kwargs):

        w = self.attrs._w
        wr = self.attrs._wr
        b = self.attrs._b

        u = self.attrs._u
        s = tanh(self.attrs._state)
        ps = self.attrs._pstate

        drt = context.restore(wr, get_gpu(u).zeros_like_me())
        dou = context.restore(w, get_gpu(dy).zeros_like_me())
        pfg = self.attrs.get("_pfgate", get_gpu(u).zeros_like_me())

        e = get_gpu(dy)

        dr, dou_n = (get_gpu(a).empty_like_me() for a in (drt, dou))

        cu.culstm_backward(*map(get_gpu, (u, dr, s, ps, e, pfg, dou, dou_n)))

        dx = dot(dr, w.T)

        context.store(wr, dr)
        context.store(w, dou_n)

        if isinstance(self.attrs._x, Node):
            self.attrs._x._update_diff(context, dx)

        if isinstance(w, Node):
            w._update_diff(context, dot(self.attrs._x.T, dr))

        if isinstance(wr, Node):
            wr._update_diff(context, dot(self.T, drt))

        if isinstance(b, Node):
            b._update_diff(context, sum(dr, axis=0))

        if isinstance(self.attrs._pz, Node):
            self.attrs._pz._update_diff(context, dot(dr, wr.T))


class Lstm(Parametrized):
    '''Long short time memory [lstm]_ .
    Lstm object has 8 weights and 4 biases parameters to learn.

    Weights applied to the input of the input gate, forget gate and output gate.
    :math:`W_{ij}, Wgi_{ij}, Wgf_{ij}, Wgo_{ij}`

    Weights applied to the recuurent input of the input gate, forget gate and output gate.
    :math:`R_{ij}, Rgi_{ij}, Rgf_{ij}, Rgo_{ij}`

    .. math::
        u^t_{i} &= \sum_{j = 0}^{J-1} W_{ij}x^t_{j} +
            \sum_{k = 0}^{K-1} R_{ik}y^{t-1}_{k} + b_i \\\\
        gi^t_{i} &= \sum_{j = 0}^{J-1} Wgi_{ij}x^t_{j} +
                \sum_{k = 0}^{K-1} Rgi_{ik}y^{t-1}_{k} + bi_i \\\\
        gf^t_{i} &= \sum_{j = 0}^{J-1} Wgfi_{ij}x^t_{j} +
                \sum_{k = 0}^{K-1} Rgf_{ik}y^{t-1}_{k} + bi_f \\\\
        go^t_{i} &= \sum_{j = 0}^{J-1} Wgo_{ij}x^t_{j} +
                \sum_{k = 0}^{K-1} Rgo_{ik}y^{t-1}_{k} + bi_o \\\\
        s^t_i &= sigmoid(gi^t_{i})tanh(u^t_{i}) + s^{t-1}_isigmoid(gf^t_{i}) \\\\
        y^t_{i} &= go^t_{i}tanh(s^t_{i})

    If the argument ``input_size`` is passed, this layers' weight is initialized
    in the __init__ function.
    Otherwise, the weight is initialized in its first forward calculation.

    Args:
        output_size (int): Output unit size.
        input_size (int): Input unit size.
        ignore_bias (bool): If True is given, bias will not be added.
        initializer (Initializer): Initializer object for weight initialization.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>>
        >>> n, d, t = (2, 3, 4)
        >>> x = rm.Variable(np.random.rand(n, d))
        >>> layer = rm.Lstm(2)
        >>> z = 0
        >>> for i in range(t):
        ...     z += rm.sum(layer(x))
        ...
        >>> grad = z.grad()    # Backpropagation.
        >>> grad.get(x)    # Gradient of x.
        Add([[-0.01853334, -0.0585249 ,  0.01290053],
             [-0.0205425 , -0.05837972,  0.00467286]], dtype=float32)
        >>> layer.truncate()

    .. [lstm] Learning Precise Timing with LSTM Recurrent Networks
    '''

    def __init__(self, output_size, input_size=None, ignore_bias=False, initializer=GlorotNormal(), weight_decay=0):
        self._size_o = output_size
        self._ignore_bias = ignore_bias
        self._initializer = initializer
        self._weight_decay = weight_decay
        super(Lstm, self).__init__(input_size)

    def weight_initiallize(self, size_i):
        size_i = size_i[0]
        size_o = self._size_o
        bias = np.zeros((1, size_o * 4), dtype=precision)
        bias[:, size_o:size_o * 2] = 1
        self.params = {
            "w": Variable(self._initializer((size_i, size_o * 4)), auto_update=True, weight_decay=self._weight_decay),
            "wr": Variable(self._initializer((size_o, size_o * 4)), auto_update=True, weight_decay=self._weight_decay)}
        if not self._ignore_bias:
            self.params["b"] = Variable(bias, auto_update=True)

    def forward(self, x):
        ret = lstm(x, self.__dict__.get("_z", None),
                   self.__dict__.get("_state", None),
                   self.params.w,
                   self.params.wr,
                   self.params.get("b", None))
        self._z = ret
        self._state = ret._state
        return ret

    def truncate(self):
        """Truncates temporal connection."""
        self._z = None
        self._state = None


class ChainedLSTM(Lstm):
    '''
    This chained LSTM model assumes an input of shape (N, T, X) where N is batch size, T is time size and X is the data.

    The model automates the process of chaining together several LSTM calls.
    '''

    def __init__(self, *args, **kwargs):
        super(ChainedLSTM, self).__init__(*args, **kwargs)

    def forward(self, x):
        lstm_model = super(ChainedLSTM, self)
        lstm_model.truncate()
        length = x.shape[1]
        for i in range(length):
            ret = lstm_model.forward(x[:, i])
        return ret
