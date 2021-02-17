#!/usr / bin / env python
# encoding: utf - 8

import numpy as np
from renom.layers.function.utils import imncol, colnim, pad_dx, pad_image, colnw
from renom.core import Node, Variable, to_value
from renom import precision
from .parameterized import Parametrized
from renom.utility.initializer import Gaussian
import renom.cuda as cu
if cu.has_cuda():
    from renom.cuda.gpuvalue import GPUValue, get_gpu
from renom.cuda import is_cuda_active


class convnd(Node):

    def __new__(cls, x, w, b, filter=3, stride=1, padding=0):
        in_shape = x.shape[1:]
        return cls.calc_value(x, w, b, in_shape, filter, stride, padding)

    @classmethod
    def _oper_cpu(cls, x, w, b, in_shape, kernel, stride, padding):
        col = imncol(to_value(x), w, stride, padding)
        if b is not None:
            col += b
        ret = cls._create_node(col)
        ret.attrs._x = x
        ret.attrs._w = w
        ret.attrs._b = b
        ret.attrs._kernel = kernel
        ret.attrs._stride = stride
        ret.attrs._padding = padding
        return ret

    @classmethod
    def _oper_gpu(cls, x, w, b, in_shape, kernel, stride, padding):
        conv_desc = cu.ConvolutionNDescriptor(padding, stride, precision)
        filter_desc = cu.NdFilterDescriptor(w.shape, precision)

        output_shape = [x.shape[0], w.shape[0]]
        for i in range(len(x.shape[2:])):
            output_shape.append((x.shape[i + 2] + padding[i] * 2 - kernel[i]) // stride[i] + 1)
        y = GPUValue(shape=tuple(output_shape))

        with cu.cudnn_handler() as handle:
            cu.cuConvolutionForward(handle, conv_desc, filter_desc, get_gpu(x), get_gpu(w), y)
            if b is not None:
                cu.cu_add_bias(get_gpu(b), y)

        ret = cls._create_node(y)
        ret.attrs._conv_desc = conv_desc
        ret.attrs._filter_desc = filter_desc
        ret.attrs._x = x
        ret.attrs._w = w
        ret.attrs._b = b
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._x, Node):
            dx = colnim(dy, self.attrs._w, self.attrs._stride)
            self.attrs._x._update_diff(context, dx)

        if isinstance(self.attrs._w, Node):
            dw = colnw(self.attrs._x, dy, self.attrs._stride)
            self.attrs._w._update_diff(context, dw)

        if isinstance(self.attrs._b, Node):
            db = np.sum(dy, axis=tuple(
                [0, ] + [i for i in range(2, len(self.attrs._b.shape))]), keepdims=True)
            self.attrs._b._update_diff(context, db)

    def _backward_gpu(self, context, dy, **kwargs):
        dw, db, dx = (get_gpu(g).empty_like_me() if g is not None else None
                      for g in (self.attrs._w, self.attrs._b, self.attrs._x))

        with cu.cudnn_handler() as handle:
            cu.cuConvolutionBackward(handle, self.attrs._conv_desc, self.attrs._filter_desc,
                                     get_gpu(self.attrs._x), get_gpu(
                                         self.attrs._w), get_gpu(dy), dw, db, dx, **kwargs)
        if isinstance(self.attrs._w, Node):
            self.attrs._w._update_diff(context, dw, **kwargs)

        if isinstance(self.attrs._x, Node):
            self.attrs._x._update_diff(context, dx, **kwargs)

        if isinstance(self.attrs._b, Node):
            self.attrs._b._update_diff(context, db, **kwargs)


def check_input(var, length):
    if isinstance(var, tuple):
        assert len(var) is length
        var = list(var)
    elif not isinstance(var, np.ndarray):
        var = np.array(
            tuple([var for _ in range(length)]), dtype=np.int32)
    elif not var.dtype == np.int32:
        var = var.astype(np.int32)
    if length < 2 and is_cuda_active():
        length = 2
    assert len(var) is length, str(len(var)) + '/' + str(length)
    return var


class ConvNd(Parametrized):
    """Nd convolution layer.

    This class creates a convolution filter to be convolved with
    the input tensor.
    The instance of this class accepts tensors of any dimensionality and produces an output of equal
    dimensionality as the input

    At instantiation, in the case of int input, filter, padding, and stride, the shape will be symmetric.

    If the argument `input_size` is passed, this layers' weight is initialized
    in the __init__ function.
    Otherwise, the weight is initialized in its first forward calculation.

    Args:
        channel (int): The dimensionality of the output.
        filter (int): Filter size of the convolution kernel.
        padding (int): Size of the zero - padding around the image.
        stride (int): Stride - size of the convolution.
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
        >>> layer = rm.ConvNd(channel=32)
        >>> z = layer(x)
        >>> z.shape
        (10, 32, 30, 30)

    Note:
        Tensor data format is **NC(D*)**.
    """

    def __init__(self, channel=2, filter=3, padding=0, stride=1,
                 input_size=None, ignore_bias=False, initializer=Gaussian()):
        self._padding = padding
        self._stride = stride
        self._kernel = filter
        self._channel = channel
        self._initial_value = [padding, stride, filter]
        self._initializer = initializer
        self._ignore_bias = ignore_bias
        super(ConvNd, self).__init__(input_size)

    def weight_initiallize(self, input_size):
        # The first dimension is to allow different types of uncorrelated images as inputs, such as RGB information.
        # After this dimension, the image data is assumed to be meaningfully correlated.
        self._dims = len(input_size[1:])
        if is_cuda_active():
            assert self._dims < 4, "GPU Version currently only supports 2 and 3 dimensions"

        if self._dims == 1 and is_cuda_active():
            padding, stride, filter = self._initial_value
            self._kernel = np.append(filter, 1).astype(np.int32)
            self._padding = np.append(padding, 0).astype(np.int32)
            self._stride = np.append(stride, 1).astype(np.int32)

        def func(var):
            return check_input(var, self._dims)
        self._kernel, self._padding, self._stride = map(
            func, [self._kernel, self._padding, self._stride])

        assert all([s > 0 for s in input_size[1:]]), \
            "The shape of input array {} is too small. Please give an array which size is lager than 0.".format(
                input_size[1:])

        f_lst = [self._channel, input_size[0]]
        f_lst.extend(self._kernel)
        size_f = tuple(f_lst)
        size_b = tuple([1, self._channel] + [1 for _ in range(self._dims)])

        self.params = {"w": Variable(self._initializer(size_f), auto_update=True)}
        if not self._ignore_bias:
            self.params["b"] = Variable(np.zeros(size_b), auto_update=True)

    def forward(self, x):
        assert len(
            x.shape) > 2, "The dimension of input array must be grater than 3. Actual dim is {}".format(x.ndim)
        assert all([s > 0 for s in x.shape[2:]]), \
            "The shape of input array {} is too small. Please give an array which size is lager than 0.".format(
                x.shape)

        return convnd(x, self.params["w"], self.params.get("b", None), self._kernel,
                      self._stride, self._padding)


class Conv3d(Parametrized):

    '''
    Provides an interface for the ConvNd with a more familiar name

    Note:
        Tensor data format is **NCHWD**.
    '''

    def __init__(self, channel=2, filter=3, padding=0, stride=1,
                 input_size=None, ignore_bias=False, initializer=Gaussian(), weight_decay=0):
        self._padding = padding
        self._stride = stride
        self._kernel = filter
        self._channel = channel
        self._initializer = initializer
        self._ignore_bias = ignore_bias
        self._weight_decay = weight_decay
        super(Conv3d, self).__init__(input_size)

    def weight_initiallize(self, input_size):
        # The first dimension is to allow different types of uncorrelated images as inputs, such as RGB information.
        # After this dimension, the image data is assumed to be meaningfully correlated.
        self._dims = len(input_size[1:])
        assert self._dims == 3, "Conv3D expects 3 dimensions"

        def func(var):
            return check_input(var, self._dims)
        self._kernel, self._padding, self._stride = map(
            func, [self._kernel, self._padding, self._stride])

        assert all([s >= min(self._kernel) for s in input_size[1:]]), \
            "The shape of input array {} is too small. Please give an array which size is lager than 0.".format(
                input_size[1:])

        f_lst = [self._channel, input_size[0]]
        f_lst.extend(self._kernel)
        size_f = tuple(f_lst)
        size_b = tuple([1, self._channel] + [1 for _ in range(self._dims)])

        self.params = {"w": Variable(self._initializer(
            size_f), auto_update=True, weight_decay=self._weight_decay)}
        if not self._ignore_bias:
            self.params["b"] = Variable(np.zeros(size_b, dtype=precision), auto_update=True)

    def forward(self, x):
        assert len(x.shape) == 5, "The dimension of input array must be 5. Actual dim is {}".format(x.ndim)
        assert all([s >= min(self._kernel) for s in x.shape[2:]]), \
            "The shape of input array {} is too small. Please give an array which size is lager than 0.".format(
                x.shape)
        return convnd(x, self.params["w"], self.params.get("b", None), self._kernel,
                      self._stride, self._padding)
