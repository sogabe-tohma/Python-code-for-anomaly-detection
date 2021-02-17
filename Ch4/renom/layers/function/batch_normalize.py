#!/usr/bin/env python
# encoding: utf-8

from __future__ import division, print_function
import numpy as np
from numbers import Number
from renom.core import Node, Variable, to_value
from renom import precision
from renom.layers.function.parameterized import Parametrized
from renom.utility.initializer import GlorotNormal
import renom.cuda as cu
if cu.has_cuda():
    from renom.cuda.gpuvalue import GPUValue, get_gpu

BATCH_NORMALIZE_FEATUREMAP = 1
BATCH_NORMALIZE_ELEMENTWISE = 2
mode_dict = {
    "activation": BATCH_NORMALIZE_ELEMENTWISE,
    "feature": BATCH_NORMALIZE_FEATUREMAP}


class batch_normalize(Node):
    def __new__(cls, x, w, b, momentum, mov_m, mov_s, inference, mode, epsilon):
        assert inference is True or x.shape[0] > 1, "Batch Normalize expects more than" \
            + " one batch when not in inference mode."
        return cls.calc_value(x, w, b, momentum, mov_m, mov_s, inference, mode, epsilon)

    @classmethod
    def _oper_cpu(cls, x, w, b, momentum, mov_m, mov_s, inference, mode, epsilon):
        if mode == BATCH_NORMALIZE_FEATUREMAP:
            axs = (0, 2, 3)
        else:
            axs = (0, )

        if inference:
            mean = mov_m
            var = mov_s
        else:
            mean = np.mean(to_value(x), axis=axs, keepdims=True)
            var = np.var(to_value(x), axis=axs, keepdims=True)

        sq_var = 1.0 / np.sqrt(var + epsilon)
        xh = (to_value(x) - mean) * sq_var
        z = to_value(w) * xh
        if b is not None:
            z += to_value(b)

        ret = cls._create_node(z)
        ret.attrs._axs = axs
        ret.attrs._x = x
        ret.attrs._w = w
        ret.attrs._b = b
        ret.attrs._m = mean
        ret.attrs._v = sq_var
        if not inference:
            N = np.prod([x.shape[s] for s in axs])
            ret.attrs._mov_m = (1 - momentum) * mov_m + momentum * mean
            ret.attrs._mov_v = (1 - momentum) * mov_s + momentum * var * N / max(N - 1., 1.)
        return ret

    @classmethod
    def _oper_gpu(cls, x, w, b, momentum, mov_m, mov_s, inference, mode, epsilon):
        if mode == BATCH_NORMALIZE_FEATUREMAP:
            axs = 1
        else:
            axs = 0

        if b is None:
            b = get_gpu(w).zeros_like_me()
        y, mean, sq_var = (get_gpu(g).empty_like_me() for g in (x, w, w))

        if inference:
            inv_var = 1.0 / np.sqrt(to_value(mov_s) + epsilon)
            if isinstance(inv_var, Number):
                inv_var = inv_var * np.ones_like(w)

        mov_m = get_gpu(mov_m)
        mov_s = get_gpu(mov_s)
        mv_m = mov_m if isinstance(mov_m, GPUValue) else get_gpu(w).zeros_like_me()
        mv_v = mov_s if isinstance(mov_s, GPUValue) else get_gpu(w).zeros_like_me()
        with cu.cudnn_handler() as handle:
            cu.cuBatchNormalizatoinForward(handle, get_gpu(x), mv_m, mv_v, get_gpu(w), get_gpu(b),
                                           y, mean, sq_var, momentum=momentum,
                                           mode=axs, inference=inference, eps=epsilon)
        ret = cls._create_node(y)
        ret.attrs._axs = axs
        ret.attrs._x = x
        ret.attrs._w = w
        ret.attrs._b = b
        if inference:
            ret.attrs._m = mv_m
            ret.attrs._v = inv_var
        else:
            ret.attrs._m = mean
            ret.attrs._v = sq_var
            ret.attrs._mov_m = mv_m
            ret.attrs._mov_v = mv_v

        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        a = self.attrs._axs
        sq_var = self.attrs._v
        meaned = self.attrs._x - self.attrs._m
        N = np.prod([self.attrs._x.shape[s] for s in a])

        if isinstance(self.attrs._x, Node):
            dxh = dy * to_value(self.attrs._w)
            ds = np.sum(dxh * meaned * -np.power(sq_var, 3) / 2, axis=a, keepdims=True)
            du = np.sum(-dxh * sq_var, axis=a, keepdims=True)
            dx = dxh * sq_var + (ds * 2 * meaned + du) / N
            self.attrs._x._update_diff(context, dx, **kwargs)
        if isinstance(self.attrs._w, Node):
            xh = meaned * sq_var
            self.attrs._w._update_diff(context, np.sum(xh * dy, axis=a, keepdims=True), **kwargs)

        if isinstance(self.attrs._b, Node):
            self.attrs._b._update_diff(context, np.sum(dy, axis=a, keepdims=True), **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        gw, gx, gdy, gm, gv = map(get_gpu, (self.attrs._w, self.attrs._x,
                                            dy, self.attrs._m, self.attrs._v))
        dx, dw, db = (g.ones_like_me() for g in (gx, gw, gw))
        ax = self.attrs._axs

        with cu.cudnn_handler() as handle:
            cu.cuBatchNormalizatoinBackward(handle, gx, gw, gdy, gm, gv, dx, dw, db, mode=ax)

        if isinstance(self.attrs._x, Node):
            self.attrs._x._update_diff(context, dx, **kwargs)

        if isinstance(self.attrs._w, Node):
            self.attrs._w._update_diff(context, dw, **kwargs)

        if isinstance(self.attrs._b, Node):
            self.attrs._b._update_diff(context, db, **kwargs)


class BatchNormalize(Parametrized):
    """Batch normalization function [bn]_.
    This layer accelerates learning speed with reducing internal covariate shift
    and allow us to set high learning rate.

    When the forward propagation, if the argument ``inference`` is set to False this layer
    calculates moving average of mean and variance.
    Other wise the ``inference`` is set to True, this layer uses the moving average which
    calculated in the above mode.

    If the argument mode is set to 'activation', this layer normalizes prior-layer features per unit.
    Otherwise the argument mode is set to 'feature', this layer normalizes prior-layer features per channel.

    The 'feature' mode is only effective for 4D tensor input.

    If the argument `input_size` is passed, this layers' weight is initialized
    in the __init__ function.
    Otherwise, the weight is initialized in its first forward calculation.

    Args:
        input_size (int): Input unit size.
        momentum (float): Momentum coefficient for the moving average.
        mode (str): 'activation'  or 'feature'.
        epsilon (float): Small number added to avoid division by zero.
        ignore_bias (bool): If `True` is given, bias will not be added.
        initializer (Initializer): Initializer object for weight initialization.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> x = np.random.rand(3, 2)
        >>> x.shape
        (3, 2)
        >>> layer = rm.BatchNormalize(momentum=0.99)
        >>> layer(x, inference=False)
        batch_normalize([[-0.05047419,  0.00471613],
                         [-0.00887055, -0.01459344],
                         [ 0.05934474,  0.00987731]], dtype=float32)

    .. [bn] Sergey Ioffe, Christian Szegedy. Batch Normalization:
        Accelerating Deep Network Training by Reducing Internal Covariate Shift(2015)

    """
    SERIALIZED = ('_mov_mean', '_mov_std', '_epsilon', '_mode')

    def __init__(self,
                 input_size=None,
                 momentum=0.99,
                 mode="activation",
                 epsilon=1e-5,
                 ignore_bias=False,
                 initializer=GlorotNormal(),
                 weight_decay=0):

        assert momentum > 0, "The value of momentum must be lager than 0."
        self._mov_mean = 0
        self._mov_std = 0
        self._epsilon = epsilon
        self._momentum = momentum
        self._mode = mode_dict.get(mode, BATCH_NORMALIZE_ELEMENTWISE)
        self.inference = False
        self._ignore_bias = ignore_bias
        self._initializer = initializer
        self._weight_decay = weight_decay
        super(BatchNormalize, self).__init__(input_size)

    def weight_initiallize(self, input_size):
        size_i = [1, ]
        size_i.extend(input_size)
        if self._mode == BATCH_NORMALIZE_FEATUREMAP and len(size_i) > 2:
            size_i[2] = 1
            size_i[3] = 1
        self.params = {"w": Variable(self._initializer(size_i).astype(
            precision), auto_update=True, weight_decay=self._weight_decay)}
        if not self._ignore_bias:
            self.params["b"] = Variable(np.zeros(size_i, dtype=precision), auto_update=True)

    def forward(self, x):
        ret = batch_normalize(x,
                              self.params["w"],
                              self.params.get("b", None),
                              self._momentum,
                              self._mov_mean,
                              self._mov_std,
                              self.inference,
                              self._mode,
                              self._epsilon)
        self._mov_mean = ret.attrs.get("_mov_m", self._mov_mean)
        self._mov_std = ret.attrs.get("_mov_v", self._mov_std)

        return ret
