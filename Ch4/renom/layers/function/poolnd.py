import numpy as np
from renom.core import Node
from renom.layers.function.utils import imnpool, poolnim
import renom.cuda as cu
if cu.has_cuda():
    from renom.cuda.gpuvalue import GPUValue, get_gpu
from renom.cuda import is_cuda_active


class npool_base(Node):

    def __new__(cls, x, kernel, stride, padding):
        return cls.calc_value(x, kernel, stride, padding)

    def _backward_gpu(self, context, dy, **kwargs):
        dx = get_gpu(self.attrs._x).empty_like_me()
        with cu.cudnn_handler() as handle:
            cu.cuPoolingBackward(handle, self.attrs._pool_desc, get_gpu(
                self.attrs._x), get_gpu(self), get_gpu(dy), dx)
        if isinstance(self.attrs._x, Node):
            self.attrs._x._update_diff(context, dx, **kwargs)


class max_poolnd(npool_base):

    @classmethod
    def _oper_cpu(cls, x, kernel, stride, padding):
        result = imnpool(x, kernel, stride, padding, mode="max")
        ret = cls._create_node(result)
        ret.attrs._x = x
        ret.attrs._kernel = kernel
        ret.attrs._stride = stride
        ret.attrs._padding = padding
        return ret

    @classmethod
    def _oper_gpu(cls, x, karnel, stride, padding):
        pool_desc = cu.PoolingNDescriptor(karnel, padding, stride, pool_mode=0)
        output_shape = [x.shape[0], x.shape[1]]
        for i in range(len(x.shape[2:])):
            output_shape.append((x.shape[i + 2] + padding[i] * 2 - karnel[i]) // stride[i] + 1)
        y = GPUValue(shape=tuple(output_shape))
        with cu.cudnn_handler() as handle:
            cu.cuPoolingForward(handle, pool_desc, get_gpu(x), get_gpu(y))
        ret = cls._create_node(y)
        ret.attrs._pool_desc = pool_desc
        ret.attrs._kernel = karnel
        ret.attrs._stride = stride
        ret.attrs._padding = padding
        ret.attrs._x = x
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        result = poolnim(self.attrs._x, dy, self.attrs._kernel,
                         self.attrs._stride, self.attrs._padding, mode="max")
        self.attrs._x._update_diff(context, result, **kwargs)


class average_poolnd(npool_base):

    @classmethod
    def _oper_cpu(cls, x, kernel, stride, padding):
        result = imnpool(x, kernel, stride, padding, mode="average")
        ret = cls._create_node(result)
        ret.attrs._x = x
        ret.attrs._kernel = kernel
        ret.attrs._stride = stride
        ret.attrs._padding = padding
        return ret

    @classmethod
    def _oper_gpu(cls, x, karnel, stride, padding):
        pool_desc = cu.PoolingNDescriptor(karnel, padding, stride, pool_mode=1)
        output_shape = [x.shape[0], x.shape[1]]
        for i in range(len(x.shape[2:])):
            output_shape.append((x.shape[i + 2] + padding[i] * 2 - karnel[i]) // stride[i] + 1)
        y = GPUValue(shape=tuple(output_shape))
        with cu.cudnn_handler() as handle:
            cu.cuPoolingForward(handle, pool_desc, get_gpu(x), get_gpu(y))
        ret = cls._create_node(y)
        ret.attrs._pool_desc = pool_desc
        ret.attrs._kernel = karnel
        ret.attrs._stride = stride
        ret.attrs._padding = padding
        ret.attrs._x = x
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        dx = poolnim(self.attrs._x, dy, self.attrs._kernel,
                     self.attrs._stride, self.attrs._padding, mode="average")
        self.attrs._x._update_diff(context, dx, **kwargs)


def check_input(var, length):
    if isinstance(var, tuple):
        assert len(var) is length
        var = list(var)
    elif not isinstance(var, np.ndarray):
        var = np.array(
            tuple([var for _ in range(length)]), dtype=np.int32)
    elif not var.dtype == np.int32:
        var = var.astype(np.int32)
    if length < 2:
        length = 2
    assert len(var) is length
    return var


class NPoolBase:

    def __init__(self, kernel=3, padding=0, stride=1):
        self._padding = padding
        self._stride = stride
        self._kernel = kernel
        self._dims = None

    def __call__(self, x):
        dims = len(x.shape[2:])
        if self._dims is None:
            if dims < 2:
                dims = 2
            self._dims = dims
        if is_cuda_active():
            assert self._dims < 4, "GPU Version can only 1, 2 and 3 dimensions"

        if self._dims == 1:
            self._kernel = np.append(self._kernel, 1).astype(np.int32)
            self._padding = np.append(self._padding, 0).astype(np.int32)
            self._stride = np.append(self._stride, 1).astype(np.int32)

        def func(var):
            return check_input(var, self._dims)
        self._padding, self._stride, self._kernel = map(
            func, [self._padding, self._stride, self._kernel])

        assert len(
            x.shape) >= 3, "The dimension of input array must be greater than 3. Actual dim is {}".format(x.ndim)
        assert all([s > 0 for s in x.shape[2:]]), \
            "The shape of input array {} is too small. Please give an array which size is lager than 0.".format(
                x.shape)
        return self.forward(x)


class Pool3Base(NPoolBase):

    def __call__(self, x):
        dims = len(x.shape[2:])
        if is_cuda_active():
            assert dims == 3, "Pool 3D expects 3 dimensions"
        super(Pool3Base, self).__call__(x)


class MaxPoolNd(NPoolBase):
    def forward(self, x):
        return max_poolnd(x, self._kernel, self._stride, self._padding)


class MaxPool3d(Pool3Base):
    def forward(self, x):
        return max_poolnd(x, self._kernel, self._stride, self._padding)


class AveragePoolNd(NPoolBase):
    def forward(self, x):
        return average_poolnd(x, self._kernel, self._stride, self._padding)


class AveragePool3d(Pool3Base):
    def forward(self, x):
        return average_poolnd(x, self._kernel, self._stride, self._padding)
