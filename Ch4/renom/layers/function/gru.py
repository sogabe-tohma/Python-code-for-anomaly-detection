#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
from renom.layers.activation.sigmoid import sigmoid
from renom.layers.activation.tanh import tanh
from renom.core import Node, Variable, GetItem
from renom import precision
from renom.operation import dot, sum, concat
from renom.utility.initializer import GlorotNormal
from .parameterized import Parametrized
import renom.cuda as cu
if cu.has_cuda():
    from renom.cuda.gpuvalue import get_gpu


def sigmoid_diff(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh_diff(x):
    return (1.0 - tanh(x) ** 2)


class gru(Node):
    '''
    @ parameters
    cls: Self variable for Python
    x: input value to the node
    pz: The previously calculated value within the same model
    w: the weights to be multiplied with the input
    u: the weights to be multiplied with the previous input
    b: the biases to be added
    '''
    def __new__(cls, x, pz, w, u, b):
        return cls.calc_value(x, pz, w, u, b)

    @classmethod
    def _oper_cpu(cls, x, pz, w, u, b):
        # Initialize Variables
        m = w.shape[1] // 3
        w_z, w_r, w_h = np.split(w, [m, m * 2, ], axis=1)
        u_z, u_r, u_h = np.split(u, [m, m * 2], axis=1)
        hminus = Variable(np.zeros((x.shape[0], w.shape[1] // 3),
                                   dtype=precision)) if pz is None else pz

        b_z, b_r, b_h = np.split(b, [m, m * 2], axis=1) if b is not None else (0, 0, 0)
        A = dot(x, w_z) + dot(hminus, u_z) + b_z
        B = dot(x, w_r) + dot(hminus, u_r) + b_r
        C = dot(x, w_h) + sigmoid(B) * dot(hminus, u_h) + b_h

        h = sigmoid(A) * hminus + (1 - sigmoid(A)) * tanh(C)

        # Store Variables for Graph
        ret = cls._create_node(h)
        ret.attrs._x = x
        ret.attrs._w = w
        ret.attrs._w_z = w_z
        ret.attrs._w_r = w_r
        ret.attrs._w_h = w_h
        ret.attrs._u = u
        ret.attrs._u_z = u_z
        ret.attrs._u_h = u_h
        ret.attrs._u_r = u_r
        ret.attrs._pz = hminus
        ret.attrs._A = A
        ret.attrs._B = B
        ret.attrs._C = C

        if b is not None:
            ret.attrs._b = b

        return ret

    @classmethod
    def _oper_gpu(cls, x, pz, w, u, b):
        # Initialize Variables
        m = w.shape[1] // 3
        hminus = Variable(np.zeros((x.shape[0], m), dtype=precision)) if pz is None else pz
        get_gpu(hminus)
        # Perform Forward Calcuations

        X = get_gpu(x)
        W = get_gpu(w)
        U = get_gpu(u)
        dotted = cu.gpuvalue.GPUValue(shape=(X.shape[0], W.shape[1]))
        cu.cublas_gemm(get_gpu(X), 0, get_gpu(W), 0, dotted)

        minusdotted = cu.gpuvalue.GPUValue(shape=(X.shape[0], W.shape[1]))
        cu.cublas_gemm(get_gpu(hminus), 0, get_gpu(U), 0, minusdotted)

        b_z, b_r, b_h = (get_gpu(b)[:, 0:m], get_gpu(b)[:, m:2 * m],
                         get_gpu(b)[:, 2 * m:3 * m]) if b is not None else (0, 0, 0)

        A = dotted[:, 0:m] + minusdotted[:, 0:m] + get_gpu(b)[:, 0:m]
        B = dotted[:, m:2 * m] + minusdotted[:, m:2 * m] + get_gpu(b)[:, m:2 * m]
        C = dotted[:, 2 * m:3 * m] + minusdotted[:, 2 * m:3 * m] * \
            B.sigmoid() + get_gpu(b)[:, 2 * m:3 * m]

        ones = A.ones_like_me()
        h = A.sigmoid() * get_gpu(hminus) + (ones - A.sigmoid()) * C.tanh()

        # Store Variables for Graph
        ret = cls._create_node(h)
        ret.attrs._x = x
        ret.attrs._w = w
        ret.attrs._b = b
        ret.attrs._u = u
        ret.attrs._pz = hminus
        ret.attrs._A = A
        ret.attrs._B = B
        ret.attrs._C = C

        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        x = self.attrs._x
        w_z = self.attrs._w_z
        w_r = self.attrs._w_r
        w_h = self.attrs._w_h
        A = self.attrs._A
        B = self.attrs._B
        C = self.attrs._C
        u_z = self.attrs._u_z
        u_h = self.attrs._u_h
        u_r = self.attrs._u_r
        hminus = self.attrs._pz
        y = dy

        dA = y * (hminus - tanh(C)) * sigmoid_diff(A)
        dC = y * (1 - sigmoid(A)) * tanh_diff(C)
        dB = dC * dot(hminus, u_h) * sigmoid_diff(B)

        # Calculate dx
        dx_z = dot(dA, w_z.T)
        dx_r = dot(dB, w_r.T)
        dx_h = dot(dC, w_h.T)
        dx = dx_z + dx_r + dx_h

        # Calculate dw
        dw_z = dot(x.T, dA)
        dw_r = dot(x.T, dB)
        dw_h = dot(x.T, dC)
        dw = np.concatenate([dw_z, dw_r, dw_h], axis=1)

        # Calculate db
        db_z = np.sum(dA, axis=0, keepdims=True)
        db_r = np.sum(dB, axis=0, keepdims=True)
        db_h = np.sum(dC, axis=0, keepdims=True)
        db = np.concatenate([db_z, db_r, db_h], axis=1)

        du_z = dot(hminus.T, dA)
        du_r = dot(hminus.T, dB)
        du_h = dot(hminus.T, dC * sigmoid(B))
        du = np.concatenate([du_z, du_r, du_h], axis=1)

        pz_z = dot(dA, u_z.T)
        pz_r = dot(dB, u_r.T)
        pz_h = dot(dC * sigmoid(B), u_h.T)

        dpz = pz_z + pz_r + pz_h + y * sigmoid(A)

        self.attrs._w._update_diff(context, dw)
        self.attrs._u._update_diff(context, du)

        if hasattr(self.attrs, "_b"):
            self.attrs._b._update_diff(context, db)

        if isinstance(self.attrs._x, Node):
            self.attrs._x._update_diff(context, dx)

        if isinstance(self.attrs._pz, Node):
            self.attrs._pz._update_diff(context, dpz)

    def prn(self, v, name='Node'):
        h = self._create_node(v)
        h.to_cpu()
        print('{:10}= {}'.format(name, h))

    def _backward_gpu(self, context, dy, **kwargs):
        x = get_gpu(self.attrs._x)
        w = get_gpu(self.attrs._w)
        b = get_gpu(self.attrs._b)
        u = get_gpu(self.attrs._u)
        hminus = get_gpu(self.attrs._pz)
        A = get_gpu(self.attrs._A)
        B = get_gpu(self.attrs._B)
        C = get_gpu(self.attrs._C)
        m = w.shape[1] // 3

        w_z, w_r, w_h = w[:, 0:m], w[:, m:2 * m], w[:, 2 * m:3 * m]
        u_z, u_r, u_h = u[:, 0:m], u[:, m:2 * m], u[:, 2 * m:3 * m]

        y = get_gpu(dy)

        def sig_diff(x):
            return x.sigmoid() * (-x.sigmoid() + 1)

        def tan_diff(x):
            return (-(x.tanh() ** 2) + 1)

        dA = y * (hminus - C.tanh()) * sig_diff(A)
        dC = y * (-A.sigmoid() + 1) * tan_diff(C)
        dB = dC * (hminus @ u_h) * sig_diff(B)

        # Calculate dx
        dx_z = dA @ w_z.T
        dx_r = dB @ w_r.T
        dx_h = dC @ w_h.T
        dx = dx_z + dx_r + dx_h

        # Calculate dw
        dw_z = x.T @ dA
        dw_r = x.T @ dB
        dw_h = x.T @ dC
        dw = w.empty_like_me()
        dw[:, m * 0:m * 1] = dw_z
        dw[:, m * 1:m * 2] = dw_r
        dw[:, m * 2:m * 3] = dw_h

        # Calculate db
        db_z = cu.cusum(dA, axis=0, keepdims=True)
        db_r = cu.cusum(dB, axis=0, keepdims=True)
        db_h = cu.cusum(dC, axis=0, keepdims=True)
        db = b.empty_like_me()
        db[:, m * 0:m * 1] = db_z
        db[:, m * 1:m * 2] = db_r
        db[:, m * 2:m * 3] = db_h

        du_z = hminus.T @ dA
        du_r = hminus.T @ dB
        du_h = hminus.T @ (dC * B.sigmoid())
        du = u.empty_like_me()
        du[:, m * 0:m * 1] = du_z
        du[:, m * 1:m * 2] = du_r
        du[:, m * 2:m * 3] = du_h

        pz_z = dA @ u_z.T
        pz_r = dB @ u_r.T
        pz_h = dC * B.sigmoid() @ u_h.T
        dpz = pz_z + pz_r + pz_h + y * A.sigmoid()

        self.attrs._w._update_diff(context, dw)
        self.attrs._u._update_diff(context, du)

        if hasattr(self.attrs, "_b"):
            self.attrs._b._update_diff(context, db)

        if isinstance(self.attrs._x, Node):
            self.attrs._x._update_diff(context, dx)

        if isinstance(self.attrs._pz, Node):
            self.attrs._pz._update_diff(context, dpz)


class Gru(Parametrized):
    '''
    Gated Recurrent Unit

    An LSTM-like RNN unit, which simplifies the LSTM unit by not including a memory core.
    This simplifies learning of the unit and reduces computational complexity, as the GRU only
    performs requires 3 input gates, compared to the 4 required by the LSTM.

    Args:
        output_size (int): Output unit size.
        input_size (int): Input unit size.
        ignore_bias (bool): If True is given, bias will not be added.
        initializer (Initializer): Initializer object for weight initialization.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> n, d, t = (2, 3, 4)
        >>> x = rm.Variable(np.random.rand(n,d))
        >>> layer = rm.Gru(2)
        >>> z = 0
        >>> for i in range(t):
        ...     z += rm.sum(layer(x))
        ...
        >>> grad = z.grad()
        >>> grad.get(x)
        Add([[-8.89559174, -0.58861321, -4.67931843],
        [-7.27466679, -0.45286781, -3.81758523]], dtype=float32)
        >>> layer.truncate()

    https://arxiv.org/pdf/1409.1259.pdf

    '''

    def __init__(self, output_size, input_size=None, ignore_bias=False, initializer=GlorotNormal(),
                 weight_decay=0):
        self._size_o = output_size
        self._initializer = initializer
        self._ignore_bias = ignore_bias
        self._weight_decay = weight_decay
        super(Gru, self).__init__(input_size)

    def weight_initiallize(self, size_i):
        size_i = size_i[0]
        size_o = self._size_o
        bias = np.zeros((1, size_o * 3), dtype=precision)
        # At this point, all connected units in the same layer will use the SAME weights
        self.params = {
            "w": Variable(self._initializer((size_i, size_o * 3)), auto_update=True, weight_decay=self._weight_decay),
            "u": Variable(self._initializer((size_o, size_o * 3)), auto_update=True, weight_decay=self._weight_decay),
        }
        if not self._ignore_bias:
            self.params["b"] = Variable(bias, auto_update=True)

    def forward(self, x):
        ret = gru(x, getattr(self, "_z", None),
                  self.params.w,
                  self.params.u,
                  self.params.get("b", None))
        self._z = ret
        return ret

    def truncate(self):
        """Truncates temporal connection."""
        self._z = None
