#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from renom.layers.function.utils import im2col, col2im, out_size, tuplize
from renom.core import Node, Variable, to_value
from renom import precision
from .parameterized import Parametrized
from renom.utility.initializer import GlorotNormal
import renom.cuda as cu
if cu.has_cuda():
    from renom.cuda.gpuvalue import GPUValue, get_gpu


class conv2d(Node):

    def __new__(cls, x, w, b, filter=3, stride=1, padding=0, dilation=1):
        filter, stride, padding, dilation = (tuplize(x)
                                             for x in (filter, stride, padding, dilation))

        in_shape = x.shape[1:]
        out_shape = [w.shape[0]]
        out_shape.extend(out_size(x.shape[2:], filter, stride, padding, dilation))
        return cls.calc_value(x, w, b, in_shape, out_shape, filter, stride, padding, dilation)

    @classmethod
    def _oper_cpu(cls, x, w, b, in_shape, out_shape, kernel, stride, padding, dilation):
        col = im2col(to_value(x),
                     out_shape[1:], kernel,
                     stride, padding, dilation)
        value = np.rollaxis(np.tensordot(col, to_value(w),
                                         ([1, 2, 3], [1, 2, 3])), 3, 1)
        if b is not None:
            value += b
        ret = cls._create_node(value)
        ret.attrs._col = col
        ret.attrs._x = x
        ret.attrs._w = w
        ret.attrs._b = b
        ret.attrs._in_shape = in_shape
        ret.attrs._out_shape = out_shape
        ret.attrs._kernel = kernel
        ret.attrs._stride = stride
        ret.attrs._padding = padding
        ret.attrs._dilation = dilation
        return ret

    @classmethod
    def _oper_gpu(cls, x, w, b, in_shape, out_shape, kernel, stride, padding, dilation):
        N = x.shape[0]
        conv_desc = cu.ConvolutionDescriptor(padding, stride, dilation, precision)
        filter_desc = cu.FilterDescriptor(w.shape, precision)

        y = GPUValue(shape=tuple([N, ] + list(out_shape)))
        with cu.cudnn_handler() as handle:
            cu.cuConvolutionForward(handle, conv_desc, filter_desc, get_gpu(x), get_gpu(w), y)
            if b is not None:
                cu.cu_add_bias(get_gpu(b), y)

        # assert type(x) is not np.ndarray

        ret = cls._create_node(y)
        ret.attrs._conv_desc = conv_desc
        ret.attrs._filter_desc = filter_desc
        ret.attrs._x = x
        ret.attrs._w = w
        ret.attrs._b = b
        ret.attrs._in_shape = in_shape
        ret.attrs._out_shape = out_shape
        ret.attrs._kernel = kernel
        ret.attrs._stride = stride
        ret.attrs._padding = padding
        ret.attrs._dilation = dilation
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        dy = to_value(dy)

        if isinstance(self.attrs._x, Node):
            dx = np.tensordot(self.attrs._w, dy, (0, 1))
            dx = np.rollaxis(dx, 3)
            dx = col2im(dx, self.attrs._in_shape[1:],
                        self.attrs._stride, self.attrs._padding, self.attrs._dilation)
            self.attrs._x._update_diff(context, dx, **kwargs)

        if isinstance(self.attrs._w, Node):
            self.attrs._w._update_diff(context, np.tensordot(
                dy, self.attrs._col, ([0, 2, 3], [0, 4, 5])), **kwargs)

        if isinstance(self.attrs._b, Node):
            self.attrs._b._update_diff(context, np.sum(dy, (0, 2, 3), keepdims=True), **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        dw, db, dx = (get_gpu(g).empty_like_me() if g is not None else None
                      for g in (self.attrs._w, self.attrs._b, self.attrs._x))

        with cu.cudnn_handler() as handle:
            if db is None:
                db = np.zeros((1, self.attrs._w.shape[0], 1, 1))
            cu.cuConvolutionBackward(handle, self.attrs._conv_desc, self.attrs._filter_desc,
                                     get_gpu(self.attrs._x), get_gpu(self.attrs._w), get_gpu(dy),
                                     get_gpu(dw), get_gpu(db), get_gpu(dx), **kwargs)
        if isinstance(self.attrs._w, Node):
            self.attrs._w._update_diff(context, dw, **kwargs)

        if isinstance(self.attrs._x, Node):
            self.attrs._x._update_diff(context, dx, **kwargs)

        if isinstance(self.attrs._b, Node):
            self.attrs._b._update_diff(context, db, **kwargs)


class Conv2d(Parametrized):
    """2d convolution layer.

    This class creates a convolution filter to be convolved with
    the input tensor.
    The instance of this class only accepts and outputs 4d tensors.

    At instantiation, in the case of int input, filter, padding, and stride, the shape will be symmetric.

    If the argument `input_size` is passed, this layers' weight is initialized
    in the __init__ function.
    Otherwise, the weight is initialized in its first forward calculation.

    Args:
        channel (int): The dimensionality of the output.
        filter (tuple,int): Filter size of the convolution kernel.
        padding (tuple,int): Size of the zero-padding around the image.
        stride (tuple,int): Stride-size of the convolution.
        dilation(tupe, int): Dilation of the convolution.
        input_size (tuple): Input unit size. This must be a tuple like (Channel, Height, Width).
        ignore_bias (bool): If `True` is given, bias will not be added.
        initializer (Initializer): Initializer object for weight initialization.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> n, c, h, w = (10, 3, 32, 32)
        >>> x = np.random.rand(n, c, h, w)
        >>> x.shape
        (10, 3, 32, 32)
        >>> layer = rm.Conv2d(channel=32)
        >>> z = layer(x)
        >>> z.shape
        (10, 32, 30, 30)

    Note:
        Tensor data format is **NCHW**.
    """

    def __init__(self,
                 channel=32,
                 filter=3,
                 padding=0,
                 stride=1,
                 dilation=1,
                 input_size=None,
                 ignore_bias=False,
                 initializer=GlorotNormal(),
                 weight_decay=0):
        self._padding, self._stride, self._kernel, self._dilation = (tuplize(x)
                                                                     for x in (padding, stride, filter, dilation))
        self._channel = channel
        self._ignore_bias = ignore_bias
        self._initializer = initializer
        self._weight_decay = weight_decay
        super(Conv2d, self).__init__(input_size)

    def weight_initiallize(self, input_size):
        size_f = (self._channel, input_size[0],
                  self._kernel[0], self._kernel[1])
        assert all([s > 0 for s in input_size[1:]]), \
            "The shape of input array {} is too small. Please give an array which size is lager than 0.".format(
                input_size[1:])
        self.params = {"w": Variable(self._initializer(size_f), auto_update=True, weight_decay=self._weight_decay)}
        if not self._ignore_bias:
            self.params["b"] = Variable(
                np.zeros((1, self._channel, 1, 1), dtype=precision), auto_update=True)

    def forward(self, x):
        assert len(x.shape) == 4, "The dimension of input array must be 4. Actual dim is {}".format(x.ndim)
        assert all([s > 0 for s in x.shape[2:]]), \
            "The shape of input array {} is small. Please give an array which size is lager than 0.".format(
                x.shape)
        return conv2d(x, self.params.w, self.params.get("b", None), self._kernel,
                      self._stride, self._padding, self._dilation)
