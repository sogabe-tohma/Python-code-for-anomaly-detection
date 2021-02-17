import contextlib
import numpy as np
from cublas import *
from libc.stdint cimport uintptr_t
import renom.cuda.base.cuda_base as cuda_base
from renom.cuda.base.cuda_base import cuCreateStream, cuDestroyStream

# This variable stores the integer value of a pointer directing towards
# the actual cublasHandle_t variable
# To get the variable, use <cublasHandle_t> _cublas_handlers[desired_id]
# Alternatively, since cublasHandle_t is an integer value itself, simply use
# _cublas_handlers[desired_id] to receive the value
_cublas_handlers = {}

cdef createCublashandler():
  cdef cublasHandle_t handle
  check(cublasCreate(&handle))
  return <uintptr_t> handle

def destroyCublas():
  cdef int i
  for i in range(len(_cublas_handlers)):
    cublasDestroy(<cublasHandle_t><uintptr_t> _cublas_handlers[i])

cdef check(cublasStatus_t status):
    if status != CUBLAS_STATUS_SUCCESS:
        raise Exception("An error has occurred in cuBlas function. Error code %d."%status)


@contextlib.contextmanager
def cublas_handler():

    cdef cublasHandle_t handle

    device_id = cuda_base.cuGetDevice()
    if device_id not in _cublas_handlers:
        check(cublasCreate_v2(&handle))
        _cublas_handlers[device_id] =  <uintptr_t>handle

    try:
        yield _cublas_handlers[device_id]
    finally:
        pass

# Scal
def cublas_scal(alf, gpu_value):
    cdef int size = gpu_value.size
    cdef uintptr_t ptr = <uintptr_t>gpu_value._ptr
    cdef uintptr_t handle = <uintptr_t> get_handle()

    cuda_base.check_heap_device(gpu_value)
    # cdef can only be defined at the function level
    cdef float alpha = <float> alf
    cdef double alpha2 = <double> alf
    if dtype == np.float32:
        check(cublasSscal(<cublasHandle_t> handle,size, &alpha, <float*>ptr, 1))
    elif gpu_value1.dtype == np.float64:
        check(cublasDscal(<cublasHandle_t> handle,size, &alpha2, <double*>ptr, 1))
    return

# AXPY
cpdef cublas_axpy(gpu_value1, gpu_value2):
    cdef int n = gpu_value1.size
    cdef uintptr_t ptr1 = <uintptr_t>gpu_value1._ptr
    cdef uintptr_t ptr2 = <uintptr_t>gpu_value2._ptr
    cdef uintptr_t handle = <uintptr_t> get_handle()

    cuda_base.check_heap_device(gpu_value1, gpu_value2)
    # cdef can only be defined at the function level
    cdef float alpha = 1.0
    cdef double alpha2 = 1.0
    if gpu_value1.dtype == np.float32:
        check(cublasSaxpy(<cublasHandle_t><uintptr_t> handle,n, &alpha, <const float*>ptr1, 1, <float*>ptr2, 1))
    elif gpu_value1.dtype == np.float64:
        check(cublasDaxpy(<cublasHandle_t><uintptr_t> handle,n, &alpha2, <const double*>ptr1, 1, <double*>ptr2, 1))
    return

cdef cudaStream_t _stream = <cudaStream_t><uintptr_t> 0
cdef cublasHandle_t _handle

def cublas_set_stream(stream):
    global _stream
    _stream = <cudaStream_t><uintptr_t> stream

'''
Receives the shared handle for cublas classes, used for parallel
execution of cublas kernels.

The function return value is the integer converted value of this pointer
To reuse this stream as a C-defined cudaStream_t variable, simply cast the
returned integer value back to cublasHandle_t
'''
cdef get_handle(idx = 0):
  # This function is a mess lol
  global _stream
  cdef uintptr_t tmp
  if not idx in _cublas_handlers:
    tmp = createCublashandler()
    _cublas_handlers[idx] = tmp
    check(cublasSetStream(<cublasHandle_t><uintptr_t>tmp, _stream))

  return _cublas_handlers[idx]


# GEMM
def cublas_gemm(gpu_value1, t1, gpu_value2, t2, gpu_value3):
    cuda_base.check_heap_device(gpu_value1, gpu_value2, gpu_value3)

    shape1 = gpu_value1.shape or (1, 1)
    shape2 = gpu_value2.shape or (1, 1)

    cdef cublasOperation_t c1 = CUBLAS_OP_T if t1 == 1 else CUBLAS_OP_N
    cdef cublasOperation_t c2 = CUBLAS_OP_T if t2 == 1 else CUBLAS_OP_N
    cdef int n = shape2[0] if t2 == 1 else shape2[1]
    cdef int m = shape1[1] if t1 == 1 else shape1[0]
    cdef int k = shape2[1] if t2 == 1 and t1 == 0 else shape2[0]
    cdef uintptr_t ptr1 = <uintptr_t>gpu_value1._ptr
    cdef uintptr_t ptr2 = <uintptr_t>gpu_value2._ptr
    cdef uintptr_t ptr3 = <uintptr_t>gpu_value3._ptr

    if len(shape1) > 2:
        raise Exception("Operation cuBlas gemm is only accept 2 dimentional matrix.")

    cdef uintptr_t handle = <uintptr_t> get_handle()
    # cdef can only be defined at the function level
    cdef float alpha = 1.0, beta = 0.0
    cdef double alpha2 = 1.0, beta2 = 0.0
    if gpu_value1.dtype == np.float32:
        check(cublasSgemm(<cublasHandle_t> handle, c2, c1, n, m, k, &alpha, <float*>ptr2, shape2[1], <float*>ptr1, shape1[1], &beta, <float*>ptr3, n))
    else:
        check(cublasDgemm(<cublasHandle_t> handle, c2, c1, n, m, k, &alpha2, <double*>ptr2, shape2[1], <double*>ptr1, shape1[1], &beta2, <double*>ptr3, n))
    return

# GEAM
def cublas_geam( a, gpu_value1, t1, b, gpu_value2, t2, gpu_value3, hand = None):
    cdef int n, m
    cdef float f_alf = <float>a
    cdef float f_bet = <float>b
    cdef double d_alf = <double>a
    cdef double d_bet = <double>b
    cdef uintptr_t ptr1 = <uintptr_t>gpu_value1._ptr
    cdef uintptr_t ptr2 = <uintptr_t>gpu_value2._ptr
    cdef uintptr_t ptr3 = <uintptr_t>gpu_value3._ptr
    cdef uintptr_t handle = <uintptr_t> get_handle() if handle is None else <uintptr_t> hand
    cdef cublasOperation_t c1 = CUBLAS_OP_T if t1 == 1 else CUBLAS_OP_N
    cdef cublasOperation_t c2 = CUBLAS_OP_T if t2 == 1 else CUBLAS_OP_N

    cuda_base.check_heap_device(gpu_value1, gpu_value2, gpu_value3)
    shape1 = gpu_value1.shape or (1, 1)
    shape2 = gpu_value2.shape or (1, 1)

    if t1 == 0:
        n = shape1[1]
        m = shape1[0]
    elif t2 == 0:
        n = shape2[1]
        m = shape2[0]
    else:
        n = shape2[0]
        m = shape2[1]

    if gpu_value1.dtype == np.float32:
        check(cublasSgeam(<cublasHandle_t><uintptr_t> handle, c1, c2, n, m, &f_alf, <const float*>ptr1, shape1[1], &f_bet,
                          <const float*>ptr2, shape2[1], <float*>ptr3, n))
    elif gpu_value1.dtype == np.float64:
        check(cublasDgeam(<cublasHandle_t><uintptr_t> handle, c1, c2, n, m, &d_alf, <const double*>ptr1, shape1[1], &d_bet,
                          <const double*>ptr2, shape2[1], <double*>ptr3, n))
    return

def cublas_transpose(handle, gpu_value1, gpu_value2):
    cuda_base.check_heap_device(gpu_value1, gpu_value2)
    cublas_geam(1.0, gpu_value1, 1, 0.0, gpu_value1, 1, gpu_value2, handle)
