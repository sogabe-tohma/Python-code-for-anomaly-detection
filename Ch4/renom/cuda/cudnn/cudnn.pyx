import contextlib
cimport numpy as cnp
import numpy as np
cimport cudnn as cd
cimport cython
from libc.stdlib cimport malloc, free
from libc.stdint cimport uintptr_t, intptr_t
from renom.config import precision
from renom.cuda.base.cuda_utils cimport _VoidPtr
from renom.cuda.base import cuda_base

cdef cudnnTensorFormat_t tensor_format = cd.cudnnTensorFormat_t.CUDNN_TENSOR_NCHW


def check(cd.cudnnStatus_t status):
    if status == cd.cudnnStatus_t.CUDNN_STATUS_SUCCESS:
        return
    else:
        error = cd.cudnnGetErrorString(status)
        raise Exception(error)


_cudnn_handlers = {}


def cudnn_set_stream(stream):
  cdef cudnnHandle_t handle

  device_id = cuda_base.cuGetDevice()
  if device_id not in _cudnn_handlers:
      check(cudnnCreate(&handle))
      _cudnn_handlers[device_id] =  <uintptr_t> handle

  handle = <cudnnHandle_t><uintptr_t> _cudnn_handlers[device_id]

  check(cudnnSetStream(handle, (<cudaStream_t><uintptr_t> stream) ))

@contextlib.contextmanager
def cudnn_handler():
    cdef cudnnHandle_t handle

    device_id = cuda_base.cuGetDevice()
    if device_id not in _cudnn_handlers:
        check(cudnnCreate(&handle))
        _cudnn_handlers[device_id] =  <uintptr_t> handle

    try:
        yield _cudnn_handlers[device_id]
    finally:
        pass


cdef data_type(dtype):
    if dtype == np.float32:
        return cd.cudnnDataType_t.CUDNN_DATA_FLOAT
    elif dtype == np.float64:
        return cd.cudnnDataType_t.CUDNN_DATA_DOUBLE
    elif dtype == np.float16:
        return cd.cudnnDataType_t.CUDNN_DATA_HALF
    else:
        raise Exception("{} is not supported type.".format(dtype))


cdef class TensorDesc(object):

    cdef cudnnTensorDescriptor_t tensor_desc

    def __init__(self, shape, dtype=precision):
        cdef int n, c, h, w
        cdef int ndims = len(shape)
        cdef int *size
        cdef int *strides

        check(cd.cudnnCreateTensorDescriptor(&(self.tensor_desc)))

        # TODO: Add safety checks for tensor parameters as well as
        # error checking tests

        if len(shape) < 5:
            n, c, h, w = list(shape) + [1] * (4 - len(shape))
            check(cd.cudnnSetTensor4dDescriptor(self.tensor_desc, tensor_format,
                                            data_type(dtype), n, c, h, w))
        else:
            size = <int *>malloc(ndims*cython.sizeof(int))
            strides = <int *>malloc(ndims*cython.sizeof(int))
            if strides is NULL or size is NULL:
                raise MemoryError()

            for i in range(ndims):
                size[i] = shape[i]
                strides[i] = np.prod(shape[ndims-i:])
            check(cd.cudnnSetTensorNdDescriptorEx(
              self.tensor_desc,
              tensor_format,
              data_type(dtype),
              ndims,
              size))
            free(size)
            free(strides)


    def __del__(self):
        if self.tensor_desc:
            check(cd.cudnnDestroyTensorDescriptor(self.tensor_desc))
            self.tensor_desc = NULL

    def __int__(self):
        return <uintptr_t>self.tensor_desc


cdef getTensorDescriptor(desc):
    cdef int n, c, h, w, ns, cs, hs, ws
    cdef cudnnDataType_t dtype
    cdef cudnnTensorDescriptor_t c_desc = <cudnnTensorDescriptor_t> <uintptr_t> desc
    cudnnGetTensor4dDescriptor(
        c_desc,
        & dtype,    # image data type
        & n,        # number of inputs (batch size)
        & c,        # number of input feature maps
        & h,        # height of input section
        & w,        # width of input section
        & ns,
        & cs,
        & hs,
        & ws)
    return n, c, h, w, ns, cs, hs, ws, <int> dtype


cdef class BaseConvolutionDescriptor:
  cdef cudnnConvolutionDescriptor_t conv_desc

  def __del__(self):
      if self.conv_desc:
          check(cudnnDestroyConvolutionDescriptor(self.conv_desc))
          self.conv_desc = NULL

  def __int__(self):
      return <uintptr_t>self.conv_desc

cdef class ConvolutionNDescriptor(BaseConvolutionDescriptor):
    def __init__(self, cnp.ndarray padding, cnp.ndarray stride, dtype):
        cdef int dimensions, pad_h, pad_w, u, v, upscalex, upscaley
        cdef cudnnConvolutionMode_t mode;
        dimensions = len(padding)
        mode = CUDNN_CONVOLUTION;
        cdef int * padArray = <int *><uintptr_t>padding.data
        cdef int * strideArray = <int *><uintptr_t>stride.data
        cdef int * upscaleArray = <int *> malloc(sizeof(int) * dimensions)
        cdef int i
        for i in range(dimensions):
          upscaleArray[i] = 1
        #  assert padArray[i] >= 0, "Padding had negative value: {}".format(padArray[i])
        #  assert strideArray[i] > 0, "Stride had negative or zero value: {}".format(strideArray[i])
        #  assert upscaleArray[i] > 0, "Upscale had negative or zero value: {}".format(upscaleArray[i])

        check(cudnnCreateConvolutionDescriptor(&(self.conv_desc)))
        check(cudnnSetConvolutionNdDescriptor(
            self.conv_desc, dimensions, padArray, strideArray, upscaleArray, mode, data_type(dtype)))
        free(upscaleArray)


cdef class ConvolutionDescriptor(BaseConvolutionDescriptor):

    def __init__(self, padding, stride, dilation, dtype):
        cdef int pad_h, pad_w, u, v, upscalex, upscaley
        cdef cudnnConvolutionMode_t mode;
        mode = CUDNN_CONVOLUTION;
        pad_h, pad_w = padding
        u, v = stride
        upscalex, upscaley = dilation

        check(cudnnCreateConvolutionDescriptor(&(self.conv_desc)))
        check(cudnnSetConvolution2dDescriptor(
            self.conv_desc, pad_h, pad_w, u, v, upscalex, upscaley, mode, data_type(dtype)))


cdef class GroupConvolutionDescriptor(BaseConvolutionDescriptor):

    def __init__(self, conv_desc, groups):
        cdef int groupcount
        cdef cudnnConvolutionDescriptor_t convDesc = <cudnnConvolutionDescriptor_t> <uintptr_t> conv_desc
        groupcount = groups

        check(cudnnSetConvolutionGroupCount(convDesc, groupcount))


cdef class PoolingNDescriptor:
    cdef cudnnPoolingDescriptor_t pool_desc

    def __init__(self, cnp.ndarray filter, cnp.ndarray padding, cnp.ndarray stride, pool_mode):
        cdef cudnnPoolingMode_t mode = cudnnPoolingMode_t.CUDNN_POOLING_MAX if pool_mode == 0 else \
                                            cudnnPoolingMode_t.CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
        cdef cudnnNanPropagation_t nan_prop = cudnnNanPropagation_t.CUDNN_NOT_PROPAGATE_NAN

        cdef int dimensions = len(filter)
        cdef int * filterArray = <int *><uintptr_t>filter.data
        cdef int * padArray = <int *><uintptr_t>padding.data
        cdef int * strideArray = <int *><uintptr_t>stride.data

        check(cudnnCreatePoolingDescriptor(& self.pool_desc))
        check(cudnnSetPoolingNdDescriptor(
            self.pool_desc, mode, nan_prop, dimensions, filterArray, padArray, strideArray))

    def __del__(self):
        if self.conv_desc:
            check(cudnnDestroyPoolingDescriptor(self.pool_desc))
            self.pool_desc= NULL

    def __int__(self):
        return <uintptr_t>self.pool_desc

cdef class PoolingDescriptor:
    cdef cudnnPoolingDescriptor_t pool_desc

    def __init__(self, filter, padding, stride, pool_mode):
        cdef int pad_h, pad_w, u, v, upscalex, upscaley
        cdef cudnnPoolingMode_t mode = cudnnPoolingMode_t.CUDNN_POOLING_MAX if pool_mode == 0 else \
                                            cudnnPoolingMode_t.CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
        cdef cudnnNanPropagation_t nan_prop = cudnnNanPropagation_t.CUDNN_NOT_PROPAGATE_NAN

        w, h = filter
        pad_h, pad_w = padding
        u, v = stride

        check(cudnnCreatePoolingDescriptor(& self.pool_desc))
        check(cudnnSetPooling2dDescriptor(
            self.pool_desc, mode, nan_prop, w, h, pad_h, pad_w, u, v))

    def __del__(self):
        if self.conv_desc:
            check(cudnnDestroyPoolingDescriptor(self.pool_desc))
            self.pool_desc= NULL

    def __int__(self):
        return <uintptr_t>self.pool_desc


cdef class LRNDescriptor:
    cdef cudnnLRNDescriptor_t lrn_desc

    def __init__(self, n, a, b, k):
        cdef cudnnConvolutionMode_t mode

        check(cudnnCreateLRNDescriptor(&self.lrn_desc))
        check(cudnnSetLRNDescriptor(
            self.lrn_desc, n, a, b, k))

    def __del__(self):
        if self.lrn_desc:
            check(cudnnDestroyLRNDescriptor(self.lrn_desc))
            self.lrn_desc= NULL

    def __int__(self):
        return <uintptr_t>self.lrn_desc


def cuPoolingForward(handle, pool_desc, x, y):

    cdef cudnnHandle_t handler = <cd.cudnnHandle_t> <uintptr_t> handle

    cdef _VoidPtr alf = _VoidPtr(np.array([1.0], dtype=x.dtype))
    cdef _VoidPtr bt = _VoidPtr(np.array([0.0], dtype=x.dtype))

    cdef TensorDesc xDesc = TensorDesc(x.shape, dtype=x.dtype)
    cdef TensorDesc yDesc = TensorDesc(y.shape, dtype=y.dtype)

    cuda_base.check_heap_device(x, y)
    check(cudnnPoolingForward(
        handler,
        <cudnnPoolingDescriptor_t> <uintptr_t> pool_desc,
        alf.ptr,
        xDesc.tensor_desc,
        <const void *> <uintptr_t> x._ptr,
        bt.ptr,
        yDesc.tensor_desc,
        <void *> <uintptr_t> y._ptr))


def cuPoolingBackward(handle, pool_desc, x, y, dy, dx):

    cdef cudnnHandle_t handler = <cd.cudnnHandle_t> <uintptr_t> handle

    cdef _VoidPtr alf = _VoidPtr(np.array([1.0], dtype=x.dtype))
    cdef _VoidPtr bt = _VoidPtr(np.array([0.0], dtype=x.dtype))
    cdef TensorDesc xDesc = TensorDesc(x.shape, dtype=x.dtype)
    cdef TensorDesc yDesc = TensorDesc(y.shape, dtype=y.dtype)

    cuda_base.check_heap_device(x, y, dy, dx)
    check(cudnnPoolingBackward(
        handler,
        <cudnnPoolingDescriptor_t> <uintptr_t> pool_desc,
        alf.ptr,
        yDesc.tensor_desc,
        <const void *> <uintptr_t> y._ptr,
        yDesc.tensor_desc,
        <const void *> <uintptr_t> dy._ptr,
        xDesc.tensor_desc,
        <const void *> <uintptr_t> x._ptr,
        bt.ptr,
        xDesc.tensor_desc,
        <void *> <uintptr_t> dx._ptr))

cdef class BaseFilterDescriptor:
  cdef cudnnFilterDescriptor_t filter_desc

  def __init__(self, shape, dtype):
      pass

  def __del__(self):
      if self.filter_desc:
          check(cudnnDestroyFilterDescriptor(self.filter_desc))
          self.filter_desc= NULL

  def __int__(self):
      return <uintptr_t>self.filter_desc

cdef class NdFilterDescriptor(BaseFilterDescriptor):

    def __init__(self, shape, dtype):
        cdef cudnnFilterDescriptor_t filter_desc
        cdef int dimensions = len(shape), i
        cdef int * dimensionArray = <int *> malloc(sizeof(int)*len(shape))
        for i in range(dimensions):
          dimensionArray[i] = <int> shape[i]

        check(cudnnCreateFilterDescriptor(&self.filter_desc))
        check(cudnnSetFilterNdDescriptor(self.filter_desc, data_type(
            dtype), tensor_format, dimensions, dimensionArray))


cdef class FilterDescriptor:
    cdef cudnnFilterDescriptor_t filter_desc

    def __init__(self, shape, dtype):
        cdef cudnnFilterDescriptor_t filter_desc
        cdef int k, c, h, w
        k, c, h, w = list(shape) + [1] * (4 - len(shape))
        check(cudnnCreateFilterDescriptor(&self.filter_desc))
        check(cudnnSetFilter4dDescriptor(self.filter_desc, data_type(
            dtype), tensor_format, k, c, h, w))

    def __del__(self):
        if self.filter_desc:
            check(cudnnDestroyFilterDescriptor(self.filter_desc))
            self.filter_desc= NULL

    def __int__(self):
        return <uintptr_t>self.filter_desc


def cuBatchNormalizatoinForward(handle, x, mean, var, w, b, y, rm, rv, momentum=0.0, mode=None, inference=False, eps=1e-5):

    cdef cudnnHandle_t handler = <cd.cudnnHandle_t> <uintptr_t> handle

    cdef _VoidPtr alf = _VoidPtr(np.array([1.0], dtype=x.dtype))
    cdef _VoidPtr bt = _VoidPtr(np.array([0.0], dtype=x.dtype))

    cdef cudnnBatchNormMode_t md
    cdef TensorDesc xDesc = TensorDesc(x.shape, dtype=x.dtype)
    cdef TensorDesc wDesc = TensorDesc(w.shape, dtype=w.dtype)
    cdef TensorDesc yDesc = TensorDesc(y.shape, dtype=y.dtype)
    cdef void * mean_ptr = <void *> <uintptr_t> getattr(mean, "_ptr", 0)
    cdef void * var_ptr = <void *> <uintptr_t> getattr(var, "_ptr", 0)

    cdef double epsilon = eps
    cdef double exponentialAverageFactor = momentum

    cuda_base.check_heap_device(x, mean, var, w, b, y, rm, rv)

    md = cudnnBatchNormMode_t.CUDNN_BATCHNORM_SPATIAL if mode == 1 else cudnnBatchNormMode_t.CUDNN_BATCHNORM_PER_ACTIVATION

    if not inference:
        check(cudnnBatchNormalizationForwardTraining(
            handler,
            md,
            alf.ptr,
            bt.ptr,
            xDesc.tensor_desc,
            <const void *> <uintptr_t> x._ptr,
            yDesc.tensor_desc,
            <void *> <uintptr_t> y._ptr,
            wDesc.tensor_desc,
            <const void *> <uintptr_t> w._ptr,
            <const void *> <uintptr_t> b._ptr,
            exponentialAverageFactor,
            mean_ptr,
            var_ptr,
            epsilon,
            <void *> <uintptr_t> rm._ptr,
            <void *> <uintptr_t> rv._ptr))
    else:
        check(cudnnBatchNormalizationForwardInference(
            handler,
            md,
            alf.ptr,
            bt.ptr,
            xDesc.tensor_desc,
            <const void *> <uintptr_t> x._ptr,
            yDesc.tensor_desc,
            <void *> <uintptr_t> y._ptr,
            wDesc.tensor_desc,
            <const void *> <uintptr_t> w._ptr,
            <const void *> <uintptr_t> b._ptr,
            mean_ptr,
            var_ptr,
            epsilon))


def cuBatchNormalizatoinBackward(handle, x, w, dy, saved_mean, saved_var, dx, dw, db, mode=None):

    cdef cudnnHandle_t handler = <cd.cudnnHandle_t> <uintptr_t> handle

    cdef _VoidPtr alf = _VoidPtr(np.array([1.0], dtype=x.dtype))
    cdef _VoidPtr bt = _VoidPtr(np.array([0.0], dtype=x.dtype))

    cdef cudnnBatchNormMode_t md
    cdef TensorDesc xDesc = TensorDesc(x.shape, dtype=x.dtype)
    cdef TensorDesc dwDesc = TensorDesc(dw.shape, dtype=dw.dtype)
    cdef TensorDesc dyDesc = TensorDesc(dy.shape, dtype=dy.dtype)
    cdef double epsilon = 1e-5

    md = cudnnBatchNormMode_t.CUDNN_BATCHNORM_SPATIAL if mode == 1 else cudnnBatchNormMode_t.CUDNN_BATCHNORM_PER_ACTIVATION

    cuda_base.check_heap_device(x, w, dy, saved_mean, saved_var, dx, dw, db)
    check(cudnnBatchNormalizationBackward(
        handler,
        md,
        alf.ptr,
        bt.ptr,
        alf.ptr,
        bt.ptr,
        xDesc.tensor_desc,
        <const void *> <uintptr_t> x._ptr,
        dyDesc.tensor_desc,
        <const void *> <uintptr_t> dy._ptr,
        xDesc.tensor_desc,
        <void *> <uintptr_t> dx._ptr,
        dwDesc.tensor_desc,
        <const void *> <uintptr_t> w._ptr,
        <void *> <uintptr_t> dw._ptr,
        <void *> <uintptr_t> db._ptr,
        epsilon,
        <const void *> <uintptr_t> saved_mean._ptr,
        <const void *> <uintptr_t> saved_var._ptr))


def cuGetConvolutionFwdAlgo(handle, conv_desc, filter_desc, x, y):

    cdef cudnnHandle_t handler = <cd.cudnnHandle_t> <uintptr_t> handle
    cdef cudnnConvolutionFwdAlgo_t algo
    cdef cudnnConvolutionFwdPreference_t pref = cudnnConvolutionFwdPreference_t.CUDNN_CONVOLUTION_FWD_NO_WORKSPACE
    cdef TensorDesc xDesc = TensorDesc(x.shape, dtype=x.dtype)
    cdef cudnnFilterDescriptor_t wDesc = <cudnnFilterDescriptor_t> <uintptr_t> filter_desc
    cdef TensorDesc yDesc = TensorDesc(y.shape, dtype=y.dtype)
    cdef cudnnConvolutionDescriptor_t convDesc = <cudnnConvolutionDescriptor_t> <uintptr_t> conv_desc

    check(cudnnGetConvolutionForwardAlgorithm(
        handler,
        xDesc.tensor_desc,
        wDesc,
        convDesc,
        yDesc.tensor_desc,
        pref,
        0,
        & algo))
    return <uintptr_t> algo


def cuConvolutionForward(handle, conv_desc, filter_desc, x, w, y):

    cdef cudnnHandle_t handler = <cd.cudnnHandle_t> <uintptr_t> handle

    cdef _VoidPtr alf = _VoidPtr(np.array([1.0], dtype=x.dtype))
    cdef _VoidPtr bt = _VoidPtr(np.array([0.0], dtype=x.dtype))

    cdef TensorDesc xDesc = TensorDesc(x.shape, dtype=x.dtype)
    cdef TensorDesc yDesc = TensorDesc(y.shape, dtype=y.dtype)
    # cdef cudnnConvolutionFwdAlgo_t algo = <cudnnConvolutionFwdAlgo_t><uintptr_t>cuGetConvolutionFwdAlgo(handle, conv_desc, filter_desc, x, y)
    # output of CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM is not deterministic
    cdef cudnnConvolutionFwdAlgo_t algo = cudnnConvolutionFwdAlgo_t.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
    cdef int workSpace = 0

    cuda_base.check_heap_device(x, w, y)
    check(cudnnConvolutionForward(
        handler,
        alf.ptr,
        xDesc.tensor_desc,
        <const void *> <uintptr_t> x._ptr,
        <cudnnFilterDescriptor_t> <uintptr_t> filter_desc,
        <void *> <uintptr_t> w._ptr,
        <cudnnConvolutionDescriptor_t> <uintptr_t> conv_desc,
        algo,
        <void *>workSpace,
        0,
        bt.ptr,
        yDesc.tensor_desc,
        <void *> <uintptr_t> y._ptr))


def cuConvolutionBackward(handle, conv_desc, filter_desc, x, w, dy, dw, db, dx):
    if db is None:
        cuda_base.check_heap_device(x, w, dy, dw, dx)
    else:
        cuda_base.check_heap_device(x, w, dy, dw, db, dx)

    cdef _VoidPtr alf = _VoidPtr(np.array([1.0], dtype=x.dtype))
    cdef _VoidPtr bt = _VoidPtr(np.array([0.0], dtype=x.dtype))

    cdef cudnnHandle_t handler = <cd.cudnnHandle_t> <uintptr_t> handle
    cdef TensorDesc xDesc = TensorDesc(x.shape, dtype=x.dtype)
    cdef TensorDesc dyDesc = TensorDesc(dy.shape, dtype=dy.dtype)
    cdef TensorDesc dbDesc

    if db is not None:
        dbDesc = TensorDesc(db.shape, dtype=db.dtype)

    cdef cudnnConvolutionBwdFilterAlgo_t algo_filter = cudnnConvolutionBwdFilterAlgo_t.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0
    cdef cudnnConvolutionBwdDataAlgo_t algo_data = cudnnConvolutionBwdDataAlgo_t.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0
    cdef int workSpace = 0
    check(cudnnConvolutionBackwardFilter(
        handler,
        alf.ptr,
        xDesc.tensor_desc,
        <const void *> <uintptr_t> x._ptr,
        dyDesc.tensor_desc,
        <const void *> <uintptr_t> dy._ptr,
        <cudnnConvolutionDescriptor_t> <uintptr_t> conv_desc,
        algo_filter,
        <void *>workSpace,
        0,
        bt.ptr,
        <cudnnFilterDescriptor_t> <uintptr_t> filter_desc,
        <void *> <uintptr_t> dw._ptr))
    check(cudnnConvolutionBackwardData(
        handler,
        alf.ptr,
        <cudnnFilterDescriptor_t> <uintptr_t> filter_desc,
        <void *> <uintptr_t> w._ptr,
        dyDesc.tensor_desc,
        <const void *> <uintptr_t> dy._ptr,
        <cudnnConvolutionDescriptor_t> <uintptr_t> conv_desc,
        algo_data,
        <void *>workSpace,
        0,
        bt.ptr,
        xDesc.tensor_desc,
        <void *> <uintptr_t> dx._ptr))

    if db is not None:
        check(cudnnConvolutionBackwardBias(
            handler,
            alf.ptr,
            dyDesc.tensor_desc,
            <const void *> <uintptr_t> dy._ptr,
            bt.ptr,
            dbDesc.tensor_desc,
            <void *> <uintptr_t> db._ptr))


def cuConvolutionBackwardData(handle, conv_desc, filter_desc, w, dy, dx):

    cdef _VoidPtr alf = _VoidPtr(np.array([1.0], dtype=w.dtype))
    cdef _VoidPtr bt = _VoidPtr(np.array([0.0], dtype=w.dtype))

    cdef cudnnHandle_t handler = <cd.cudnnHandle_t> <uintptr_t> handle
    cdef TensorDesc xDesc = TensorDesc(dx.shape, dtype=dx.dtype)
    cdef TensorDesc dyDesc = TensorDesc(dy.shape, dtype=dy.dtype)
    cdef cudnnConvolutionBwdDataAlgo_t algo_data = cudnnConvolutionBwdDataAlgo_t.CUDNN_CONVOLUTION_BWD_DATA_ALGO_0
    cdef int workSpace = 0

    cuda_base.check_heap_device(w, dy, dx)
    check(cudnnConvolutionBackwardData(
        handler,
        alf.ptr,
        <cudnnFilterDescriptor_t> <uintptr_t> filter_desc,
        <void *> <uintptr_t> w._ptr,
        dyDesc.tensor_desc,
        <const void *> <uintptr_t> dy._ptr,
        <cudnnConvolutionDescriptor_t> <uintptr_t> conv_desc,
        algo_data,
        <void *>workSpace,
        0,
        bt.ptr,
        xDesc.tensor_desc,
        <void *> <uintptr_t> dx._ptr))


def cuConvolutionBackwardFilter(handle, conv_desc, filter_desc, x, dy, dw):

    cdef _VoidPtr alf = _VoidPtr(np.array([1.0], dtype=x.dtype))
    cdef _VoidPtr bt = _VoidPtr(np.array([0.0], dtype=x.dtype))

    cdef cudnnHandle_t handler = <cd.cudnnHandle_t> <uintptr_t> handle
    cdef TensorDesc xDesc = TensorDesc(x.shape, dtype=x.dtype)
    cdef TensorDesc dyDesc = TensorDesc(dy.shape, dtype=dy.dtype)
    cdef cudnnConvolutionBwdFilterAlgo_t algo_filter = cudnnConvolutionBwdFilterAlgo_t.CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0
    cdef int workSpace = 0

    cuda_base.check_heap_device(x, dy, dw)
    check(cudnnConvolutionBackwardFilter(
        handler,
        alf.ptr,
        xDesc.tensor_desc,
        <const void *> <uintptr_t> x._ptr,
        dyDesc.tensor_desc,
        <const void *> <uintptr_t> dy._ptr,
        <cudnnConvolutionDescriptor_t> <uintptr_t> conv_desc,
        algo_filter,
        <void *>workSpace,
        0,
        bt.ptr,
        <cudnnFilterDescriptor_t> <uintptr_t> filter_desc,
        <void *> <uintptr_t> dw._ptr))


def cuConvolutionBackwardBias(handle, dy, db):

    cdef _VoidPtr alf = _VoidPtr(np.array([1.0], dtype=dy.dtype))
    cdef _VoidPtr bt = _VoidPtr(np.array([0.0], dtype=dy.dtype))

    cdef cudnnHandle_t handler = <cd.cudnnHandle_t> <uintptr_t> handle
    cdef TensorDesc dyDesc = TensorDesc(dy.shape, dtype=dy.dtype)
    cdef TensorDesc dbDesc = TensorDesc(db.shape, dtype=db.dtype)

    cuda_base.check_heap_device(dy, db)
    check(cudnnConvolutionBackwardBias(
        handler,
        alf.ptr,
        dyDesc.tensor_desc,
        <const void *> <uintptr_t> dy._ptr,
        bt.ptr,
        dbDesc.tensor_desc,
        <void *> <uintptr_t> db._ptr))


def cuSoftmaxForward(handle, x, y, mode=0):

    cdef _VoidPtr a = _VoidPtr(np.array([1.0], dtype=x.dtype))
    cdef _VoidPtr b = _VoidPtr(np.array([0.0], dtype=x.dtype))

    cdef cudnnHandle_t handler = <cd.cudnnHandle_t> <uintptr_t> handle
    cdef TensorDesc yDesc = TensorDesc(y.shape, dtype=y.dtype)
    cdef cd.cudnnSoftmaxMode_t md = cd.cudnnSoftmaxMode_t.CUDNN_SOFTMAX_MODE_CHANNEL if mode == 1 else cd.cudnnSoftmaxMode_t.CUDNN_SOFTMAX_MODE_INSTANCE

    cuda_base.check_heap_device(x, y)
    check(cd.cudnnSoftmaxForward(
        handler,
        cd.cudnnSoftmaxAlgorithm_t.CUDNN_SOFTMAX_ACCURATE,
        md,
        <const void *> a.ptr,
        yDesc.tensor_desc,
        <const void *> <uintptr_t> x._ptr,
        <const void *> b.ptr,
        yDesc.tensor_desc,
        <void *> <uintptr_t> y._ptr))


def cuSoftmaxBackward(handle, y, dy, dx, mode=0):

    cdef _VoidPtr a = _VoidPtr(np.array([1.0], dtype=dy.dtype))
    cdef _VoidPtr b = _VoidPtr(np.array([0.0], dtype=dy.dtype))

    cdef cudnnHandle_t handler = <cd.cudnnHandle_t> <uintptr_t> handle
    cdef TensorDesc dyDesc = TensorDesc(dy.shape, dtype=dy.dtype)
    cdef cd.cudnnSoftmaxMode_t md = cd.cudnnSoftmaxMode_t.CUDNN_SOFTMAX_MODE_CHANNEL if mode == 1 else cd.cudnnSoftmaxMode_t.CUDNN_SOFTMAX_MODE_INSTANCE

    cuda_base.check_heap_device(y, dx, dy)
    check(cd.cudnnSoftmaxBackward(
        handler,
        cd.cudnnSoftmaxAlgorithm_t.CUDNN_SOFTMAX_ACCURATE,
        md,
        <const void *> a.ptr,
        dyDesc.tensor_desc,
        <const void *> <uintptr_t> y._ptr,
        dyDesc.tensor_desc,
        <const void *> <uintptr_t> dy._ptr,
        <const void *> b.ptr,
        dyDesc.tensor_desc,
        <void *> <uintptr_t> dx._ptr))


def cuLocalResponseNormalizationForward(handle, lrn_desc, x, y):

    cdef cudnnHandle_t handler = <cd.cudnnHandle_t> <uintptr_t> handle
    cdef cudnnLRNMode_t mode = cudnnLRNMode_t.CUDNN_LRN_CROSS_CHANNEL_DIM1
    cdef TensorDesc xDesc = TensorDesc(x.shape, dtype=x.dtype)
    cdef TensorDesc yDesc = TensorDesc(y.shape, dtype=y.dtype)

    cdef _VoidPtr d = _VoidPtr(np.array([1.0], dtype=x.dtype))
    cdef _VoidPtr e = _VoidPtr(np.array([0.0], dtype=x.dtype))

    cuda_base.check_heap_device(x, y)
    check(cudnnLRNCrossChannelForward(
        handler,
        <cudnnLRNDescriptor_t> <uintptr_t> lrn_desc,
        mode,
        d.ptr,
        xDesc.tensor_desc,
        <const void *> <uintptr_t> x._ptr,
        e.ptr,
        yDesc.tensor_desc,
        <void *> <uintptr_t> y._ptr))


def cuLocalResponseNormalizationBackward(handle, lrn_desc, x, y, dx, dy):

    cdef cudnnHandle_t handler = <cd.cudnnHandle_t> <uintptr_t> handle
    cdef cudnnLRNMode_t mode = cudnnLRNMode_t.CUDNN_LRN_CROSS_CHANNEL_DIM1
    cdef TensorDesc xDesc = TensorDesc(x.shape, dtype=x.dtype)
    cdef TensorDesc yDesc = TensorDesc(y.shape, dtype=y.dtype)

    cdef _VoidPtr a = _VoidPtr(np.array([1.0], dtype=x.dtype))
    cdef _VoidPtr b = _VoidPtr(np.array([0.0], dtype=x.dtype))

    cuda_base.check_heap_device(x, y, dx, dy)
    check(cudnnLRNCrossChannelBackward(
        handler,
        <cudnnLRNDescriptor_t> <uintptr_t> lrn_desc,
        mode,
        a.ptr,
        yDesc.tensor_desc,
        <void *> <uintptr_t> y._ptr,
        yDesc.tensor_desc,
        <const void *> <uintptr_t> dy._ptr,
        xDesc.tensor_desc,
        <const void *> <uintptr_t> x._ptr,
        b.ptr,
        xDesc.tensor_desc,
        <void *> <uintptr_t> dx._ptr))
