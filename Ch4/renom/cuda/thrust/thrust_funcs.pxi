#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
関数命名規則
関数名: cu〜    (python側から呼ばれる関数)

引数名: gpu_value
"""
import numpy as np
from libc.stdint cimport uintptr_t
from libc.stdio cimport printf
from libcpp cimport bool
import renom.cuda.base.cuda_base as cuda_base
import operator
import functools
import renom.cuda

# For debug
import time


def cunegate(input, result):
    cuda_base.check_heap_device(input, result)

    cdef VALUE_TYPE * first = <VALUE_TYPE * > < uintptr_t > input._ptr
    cdef VALUE_TYPE * last = first + <size_t > input.size
    cdef VALUE_TYPE * output = <VALUE_TYPE * > < uintptr_t > result._ptr
    thrust_negate(first, last, output)


def curelu_foward(gpu_value1, gpu_value2):
    cuda_base.check_heap_device(gpu_value1, gpu_value2)

    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    thrust_relu_forward(ptr1, ptr2, size)


def curelu_backard(gpu_value1, gpu_value2):
    cuda_base.check_heap_device(gpu_value1, gpu_value2)

    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    thrust_relu_backward(ptr1, ptr2, size)


def curelu6_foward(gpu_value1, gpu_value2):
    cuda_base.check_heap_device(gpu_value1, gpu_value2)

    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    thrust_relu6_forward(ptr1, ptr2, size)


def curelu6_backard(gpu_value1, gpu_value2):
    cuda_base.check_heap_device(gpu_value1, gpu_value2)

    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    thrust_relu6_backward(ptr1, ptr2, size)


def culeaky_leru_forward(s, gpu_value1, gpu_value2):
    cuda_base.check_heap_device(gpu_value1, gpu_value2)

    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    thrust_leaky_relu_forward(< VALUE_TYPE > s, ptr1, ptr2, size);


def culeaky_leru_backward(s, gpu_value1, gpu_value2):
    cuda_base.check_heap_device(gpu_value1, gpu_value2)

    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    thrust_leaky_relu_backward(< VALUE_TYPE > s, ptr1, ptr2, size);


def cueru_forward(s, gpu_value1, gpu_value2):
    cuda_base.check_heap_device(gpu_value1, gpu_value2)

    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    thrust_elu_forward(< VALUE_TYPE > s, ptr1, ptr2, size);


def cueru_backward(s, gpu_value1, gpu_value2):
    cuda_base.check_heap_device(gpu_value1, gpu_value2)

    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    thrust_elu_backward(< VALUE_TYPE > s, ptr1, ptr2, size);


def cusoftplus_forward(gpu_value1, gpu_value2):
    cuda_base.check_heap_device(gpu_value1, gpu_value2)

    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    thrust_softplus_forward(ptr1, ptr2, size)


def cusoftplus_backward(gpu_value1, gpu_value2, gpu_value3):
    cuda_base.check_heap_device(gpu_value1, gpu_value2, gpu_value3)

    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    cdef VALUE_TYPE * ptr3 = <VALUE_TYPE * > < uintptr_t > gpu_value3._ptr
    thrust_softplus_backward(ptr1, ptr2, ptr3, size)


def cusoftsign_forward(gpu_value1, gpu_value2):
    cuda_base.check_heap_device(gpu_value1, gpu_value2)

    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    thrust_softsign_forward(ptr1, ptr2, size)


def cusoftsign_backward(gpu_value1, gpu_value2):
    cuda_base.check_heap_device(gpu_value1, gpu_value2)

    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    thrust_softsign_backward(ptr1, ptr2, size)


def cusigmoid(gpu_value1, gpu_value2):
    cuda_base.check_heap_device(gpu_value1, gpu_value2)

    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    thrust_sigmoid(ptr1, ptr2, size)


def cuhard_sigmoid_forward(gpu_value1, gpu_value2):
    cuda_base.check_heap_device(gpu_value1, gpu_value2)

    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    thrust_hard_sigmoid_forward(ptr1, ptr2, size)


def cuhard_sigmoid_backward(gpu_value1, gpu_value2):
    cuda_base.check_heap_device(gpu_value1, gpu_value2)

    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    thrust_hard_sigmoid_backward(ptr1, ptr2, size)


def cutanh(gpu_value1, gpu_value2):
    cuda_base.check_heap_device(gpu_value1, gpu_value2)

    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    thrust_tanh(ptr1, ptr2, size)


def cuhard_tanh_forward(gpu_value1, gpu_value2):
    cuda_base.check_heap_device(gpu_value1, gpu_value2)

    cdef int size = < int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = < VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = < VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    thrust_hard_tanh_forward(ptr1, ptr2, size)


def cuhard_tanh_backward(gpu_value1, gpu_value2):
    cuda_base.check_heap_device(gpu_value1, gpu_value2)

    cdef int size = < int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = < VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = < VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    thrust_hard_tanh_backward(ptr1, ptr2, size)


def cuswish_forward(s, gpu_value1, gpu_value2):
    cuda_base.check_heap_device(gpu_value1, gpu_value2)

    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    thrust_swish_forward(< VALUE_TYPE > s, ptr1, ptr2, size);


def cuswish_backward(s, gpu_value1, gpu_value2):
    cuda_base.check_heap_device(gpu_value1, gpu_value2)

    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    thrust_swish_backward(< VALUE_TYPE > s, ptr1, ptr2, size);


def cumish_forward(gpu_value1, gpu_value2):
    cuda_base.check_heap_device(gpu_value1, gpu_value2)

    cdef int size = < int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = < VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = < VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    thrust_mish_forward(ptr1, ptr2, size)


def cumish_backward(gpu_value1, gpu_value2):
    cuda_base.check_heap_device(gpu_value1, gpu_value2)

    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    thrust_mish_backward(ptr1, ptr2, size)


ctypedef void(*BINOP_FUNC)(
    VALUE_TYPE * a, VALUE_TYPE * b, VALUE_TYPE * c,
    size_t size, binop_strides * strides)


cpdef calc_strides(shape):
    cdef int shapelen = len(shape)
    if not shapelen:
        return []
    ret = [0] * (shapelen - 1) + [1]
    cdef int n
    for n in range(-1, shapelen * -1, -1):
        ret[n - 1] = shape[n] * ret[n]
    return ret


cpdef calc_int_prod(arr):
    cdef int arrlen = len(arr)
    cdef int ret = 1

    cdef int n
    for i in range(0, arrlen):
        ret *= arr[i]
    return ret


cdef bin_operation(BINOP_FUNC func, lhs, rhs, ret):

    cuda_base.check_heap_device(lhs, rhs, ret)

    if not isinstance(rhs, renom.core.GPUValue):
        rhs = renom.core.GPUValue(np.array(rhs))

    cdef binop_strides strides

    start_t = time.time()
    if lhs.shape == rhs.shape == ret.shape:
        strides.size = 1
        strides.result_strides[0] = 1
        strides.lhs_strides[0] = 1
        strides.rhs_strides[0] = 1
    else:
        ret_strides = calc_strides(ret.shape)

        lhs_strides = calc_strides(lhs.shape)
        lhs_strides = [0] * (len(ret.shape) - len(lhs.shape)) + lhs_strides

        for i, (arg, dest) in enumerate(zip(reversed(lhs.shape), reversed(ret.shape)), 1):
            if arg != dest:
                lhs_strides[i * -1] = 0

        rhs_strides = calc_strides(rhs.shape)
        rhs_strides = [0] * (len(ret.shape) - len(rhs.shape)) + rhs_strides

        for i, (arg, dest) in enumerate(zip(reversed(rhs.shape), reversed(ret.shape)), 1):
            if arg != dest:
                rhs_strides[i * -1] = 0

        strides.size = len(ret_strides)
        for i in range(strides.size):
            strides.result_strides[i] = ret_strides[i]
            strides.lhs_strides[i] = lhs_strides[i]
            strides.rhs_strides[i] = rhs_strides[i]

    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > lhs._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > rhs._ptr
    cdef VALUE_TYPE * ptr3 = <VALUE_TYPE * > < uintptr_t > ret._ptr
    size = calc_int_prod(ret.shape)

    assert strides.size < 6, "Binary operation error. Only tensors that has less than 6dims are accepted. Actual is {} dim tensor.".format(
        strides.size)

    func(ptr1, ptr2, ptr3, size, & strides)


ctypedef void(*BINOP_FUNC_NUM)(
    VALUE_TYPE * a, VALUE_TYPE b, VALUE_TYPE * c,
    size_t size)


cdef bin_operation_num(BINOP_FUNC_NUM func, lhs, rhs, ret):
    cuda_base.check_heap_device(lhs, ret)

    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > lhs._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > ret._ptr
    size = calc_int_prod(ret.shape)

    cdef VALUE_TYPE num = rhs

    func(ptr1, num, ptr2, size)


def cumul(gpu_value1, gpu_value2, gpu_value3):
    cuda_base.check_heap_device(gpu_value1, gpu_value2, gpu_value3)

    if isinstance(gpu_value2, renom.core.GPUValue):
        bin_operation(thrust_mul, gpu_value1, gpu_value2, gpu_value3)
    else:
        bin_operation_num(thrust_mul_num, gpu_value1, gpu_value2, gpu_value3)


def cuadd(gpu_value1, gpu_value2, gpu_value3):
    cuda_base.check_heap_device(gpu_value1, gpu_value2, gpu_value3)

    if isinstance(gpu_value2, renom.core.GPUValue):
        bin_operation(thrust_add, gpu_value1, gpu_value2, gpu_value3)
    else:
        bin_operation_num(thrust_add_num, gpu_value1, gpu_value2, gpu_value3)


def cusub(gpu_value1, gpu_value2, gpu_value3):
    cuda_base.check_heap_device(gpu_value1, gpu_value2, gpu_value3)

    if isinstance(gpu_value2, renom.core.GPUValue):
        bin_operation(thrust_sub, gpu_value1, gpu_value2, gpu_value3)
    else:
        bin_operation_num(thrust_sub_num, gpu_value1, gpu_value2, gpu_value3)


def cudiv(gpu_value1, gpu_value2, gpu_value3):
    cuda_base.check_heap_device(gpu_value1, gpu_value2, gpu_value3)

    if isinstance(gpu_value2, renom.core.GPUValue):
        bin_operation(thrust_div, gpu_value1, gpu_value2, gpu_value3)
    else:
        bin_operation_num(thrust_div_num, gpu_value1, gpu_value2, gpu_value3)


def curdiv(gpu_value1, gpu_value2, gpu_value3):
    cuda_base.check_heap_device(gpu_value1, gpu_value2, gpu_value3)

    if isinstance(gpu_value2, renom.core.GPUValue):
        bin_operation(thrust_rdiv, gpu_value1, gpu_value2, gpu_value3)
    else:
        bin_operation_num(thrust_rdiv_num, gpu_value1, gpu_value2, gpu_value3)


def cupow(gpu_value1, gpu_value2, gpu_value3):
    cuda_base.check_heap_device(gpu_value1, gpu_value2, gpu_value3)

    if isinstance(gpu_value2, renom.core.GPUValue):
        bin_operation(thrust_pow, gpu_value1, gpu_value2, gpu_value3)
    else:
        bin_operation_num(thrust_pow_num, gpu_value1, gpu_value2, gpu_value3)


def curpow(gpu_value1, gpu_value2, gpu_value3):
    cuda_base.check_heap_device(gpu_value1, gpu_value2, gpu_value3)

    if isinstance(gpu_value2, renom.core.GPUValue):
        bin_operation(thrust_rpow, gpu_value1, gpu_value2, gpu_value3)
    else:
        bin_operation_num(thrust_rpow_num, gpu_value1, gpu_value2, gpu_value3)


def cufill(value, gpu_value):
    cdef int size = <int > gpu_value.size
    cdef VALUE_TYPE v = <VALUE_TYPE > value
    cdef VALUE_TYPE * ptr = <VALUE_TYPE * > < uintptr_t > gpu_value._ptr

    cuda_base.check_heap_device(gpu_value)
    thrust_fill(v, ptr, size)


def culoge(gpu_value1, gpu_value2):
    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr

    cuda_base.check_heap_device(gpu_value1, gpu_value2)
    thrust_loge(ptr1, ptr2, size)


def cuexp(gpu_value1, gpu_value2):
    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    thrust_exp(ptr1, ptr2, size)


def cusqrt(gpu_value1, gpu_value2):
    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr

    cuda_base.check_heap_device(gpu_value1, gpu_value2)
    thrust_sqrt(ptr1, ptr2, size)


def cusign(gpu_value1, gpu_value2):
    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    cuda_base.check_heap_device(gpu_value1, gpu_value2)
    thrust_sign(ptr1, ptr2, size)


def cucross_entropy(gpu_value1, gpu_value2, gpu_value3):
    cdef int size = <int > gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    cdef VALUE_TYPE * ptr3 = <VALUE_TYPE * > < uintptr_t > gpu_value3._ptr

    cuda_base.check_heap_device(gpu_value1, gpu_value2, gpu_value3)
    thrust_cross_entropy(ptr1, ptr2, ptr3, size)


def cuabs_forward(gpu_value1, gpu_value2):
    cdef int size = gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr

    cuda_base.check_heap_device(gpu_value1, gpu_value2)
    thrust_abs_forward(ptr1, ptr2, size)


def cuabs_backward(gpu_value1, gpu_value2):
    cdef int size = gpu_value1.size
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr

    cuda_base.check_heap_device(gpu_value1, gpu_value2)
    thrust_abs_backward(ptr1, ptr2, size)


def cumin(value, gpu_value1, gpu_value2=None):
    cdef int size = gpu_value1.size
    cdef VALUE_TYPE * ptr1 = < VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = < VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    cdef VALUE_TYPE v = <VALUE_TYPE > value

    cuda_base.check_heap_device(gpu_value1, gpu_value2)
    thrust_min(v, ptr1, ptr2, size)


def cumax(value, gpu_value1, gpu_value2=None):
    cdef int size = gpu_value1.size
    cdef VALUE_TYPE * ptr1 = < VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * ptr2 = < VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    cdef VALUE_TYPE v = <VALUE_TYPE > value

    cuda_base.check_heap_device(gpu_value1, gpu_value2)
    thrust_max(v, ptr1, ptr2, size)


def curoi_pool2d_forward(rois, x, spatial_scale, channels, height,
                         width, outh, outw, z, augmax_data):
    cdef int N = rois.shape[0]
    cdef VALUE_TYPE * ptr_x = <VALUE_TYPE * > < uintptr_t > x._ptr
    cdef VALUE_TYPE * ptr_rois = <VALUE_TYPE * > < uintptr_t > rois._ptr
    cdef VALUE_TYPE * ptr_z = <VALUE_TYPE * > < uintptr_t > z._ptr
    cdef VALUE_TYPE * ptr_augmax_data = <VALUE_TYPE * > < uintptr_t > augmax_data._ptr
    thrust_forward_roi_pool2d(N, ptr_x, spatial_scale, channels, height,
                              width, outh, outw, ptr_rois, ptr_z, ptr_augmax_data)


def curoi_pool2d_backward(du, argmax, rois, spatial_scale, ch, h, w, outh, outw, dx):
    cdef int roi_N = rois.shape[0]
    cdef int batch_N = dx.shape[0]
    cdef VALUE_TYPE * ptr_du = <VALUE_TYPE * > < uintptr_t > du._ptr
    cdef VALUE_TYPE * ptr_argmax = <VALUE_TYPE * > < uintptr_t > argmax._ptr
    cdef VALUE_TYPE * ptr_rois = <VALUE_TYPE * > < uintptr_t > rois._ptr
    cdef VALUE_TYPE * ptr_dx = <VALUE_TYPE * > < uintptr_t > dx._ptr
    thrust_backward_roi_pool2d(roi_N, ptr_du, ptr_argmax, ptr_rois,
                               spatial_scale, batch_N, ch, h, w, outh, outw, ptr_dx)


def culstm_forward_activate(u):
    cdef int N = u.shape[0]
    cdef int M = u.shape[1]

    cdef VALUE_TYPE * ptr_u = < VALUE_TYPE * > < uintptr_t > u._ptr
    thrust_forward_lstm_activate(N, M, ptr_u)


def culstm_forward(u, s, ps, z):
    cdef int N = u.shape[0]
    cdef int M = u.shape[1]

    cdef VALUE_TYPE * ptr_u = < VALUE_TYPE * > < uintptr_t > u._ptr
    cdef VALUE_TYPE * ptr_s = < VALUE_TYPE * > < uintptr_t > s._ptr
    cdef VALUE_TYPE * ptr_ps = < VALUE_TYPE * > < uintptr_t > ps._ptr
    cdef VALUE_TYPE * ptr_z = < VALUE_TYPE * > < uintptr_t > z._ptr
    thrust_forward_lstm(N, M, ptr_u, ptr_s, ptr_ps, ptr_z)


def culstm_backward(u, du, s, ps, e, pgf, dou, dou_n):
    cdef int N = u.shape[0]
    cdef int M = u.shape[1]
    cdef VALUE_TYPE * ptr_u = < VALUE_TYPE * > < uintptr_t > u._ptr
    cdef VALUE_TYPE * ptr_du = < VALUE_TYPE * > < uintptr_t > du._ptr
    cdef VALUE_TYPE * ptr_s = < VALUE_TYPE * > < uintptr_t > s._ptr
    cdef VALUE_TYPE * ptr_ps = < VALUE_TYPE * > < uintptr_t > ps._ptr
    cdef VALUE_TYPE * ptr_e = < VALUE_TYPE * > < uintptr_t > e._ptr
    cdef VALUE_TYPE * ptr_pgf = < VALUE_TYPE * > < uintptr_t > pgf._ptr
    cdef VALUE_TYPE * ptr_dou = < VALUE_TYPE * > < uintptr_t > dou._ptr
    cdef VALUE_TYPE * ptr_dou_n = < VALUE_TYPE * > < uintptr_t > dou_n._ptr
    thrust_backward_lstm(N, M, ptr_u, ptr_du, ptr_s, ptr_ps,
                         ptr_e, ptr_pgf, ptr_dou, ptr_dou_n)


def cupeepholelstm_forward(u, wc, prestate, state, z):
    cuda_base.check_heap_device(u, prestate, state, wc, z)

    cdef int N = u.shape[0]
    cdef int M = u.shape[1]
    cdef VALUE_TYPE * ptr_u = < VALUE_TYPE * > < uintptr_t > u._ptr
    cdef VALUE_TYPE * ptr_z = < VALUE_TYPE * > < uintptr_t > z._ptr
    cdef VALUE_TYPE * ptr_ps = < VALUE_TYPE * > < uintptr_t > prestate._ptr
    cdef VALUE_TYPE * ptr_s = < VALUE_TYPE * > < uintptr_t > state._ptr
    cdef VALUE_TYPE * ptr_wc = < VALUE_TYPE * > < uintptr_t > wc._ptr
    thrust_forward_peephole_lstm(N, M, ptr_u, ptr_wc, ptr_ps, ptr_s, ptr_z)


def cupeepholelstm_backward(u, prestate, state, prefg, wc, dy, drt, dot, dr, dou, dwc):
    cuda_base.check_heap_device(u, prestate, state, prestate, wc,
                                dy, drt, dot, dou, dr, dwc)
    cdef int N = u.shape[0]
    cdef int M = u.shape[1]

    cdef VALUE_TYPE * ptr_u = < VALUE_TYPE * > < uintptr_t > u._ptr
    cdef VALUE_TYPE * ptr_ps = < VALUE_TYPE * > < uintptr_t > prestate._ptr
    cdef VALUE_TYPE * ptr_s = < VALUE_TYPE * > < uintptr_t > state._ptr
    cdef VALUE_TYPE * ptr_pfg = < VALUE_TYPE * > < uintptr_t > prefg._ptr
    cdef VALUE_TYPE * ptr_wc = < VALUE_TYPE * > < uintptr_t > wc._ptr
    cdef VALUE_TYPE * ptr_dy = < VALUE_TYPE * > < uintptr_t > dy._ptr
    cdef VALUE_TYPE * ptr_drt = < VALUE_TYPE * > < uintptr_t > drt._ptr
    cdef VALUE_TYPE * ptr_dot = < VALUE_TYPE * > < uintptr_t > dot._ptr
    cdef VALUE_TYPE * ptr_dr = < VALUE_TYPE * > < uintptr_t > dr._ptr
    cdef VALUE_TYPE * ptr_dou = < VALUE_TYPE * > < uintptr_t > dou._ptr
    cdef VALUE_TYPE * ptr_dwc = < VALUE_TYPE * > < uintptr_t > dwc._ptr
    thrust_backward_peephole_lstm(N, M, ptr_u, ptr_ps, ptr_s, ptr_pfg, ptr_wc,
                                  ptr_dy, ptr_drt, ptr_dot, ptr_dr, ptr_dou, ptr_dwc)


def cugru_forward(input, hminus, u, ABC, h):
    cdef int X = input.shape[0]
    cdef int Y = input.shape[1]
    cdef int M = input.shape[1] // 3
    cdef VALUE_TYPE * ptr_input = < VALUE_TYPE * > < uintptr_t > input._ptr
    cdef VALUE_TYPE * ptr_hminus = < VALUE_TYPE * > < uintptr_t > hminus._ptr
    cdef VALUE_TYPE * ptr_u = < VALUE_TYPE * > < uintptr_t > u._ptr
    cdef VALUE_TYPE * ptr_ABC = < VALUE_TYPE * > < uintptr_t > ABC._ptr
    cdef VALUE_TYPE * ptr_h = < VALUE_TYPE * > < uintptr_t > h._ptr
    thrust_forward_gru(X, Y, M, ptr_input, ptr_hminus, ptr_u, ptr_ABC, ptr_h)


def cugru_backward(a, b, c, d, e, f, g, h, i):
    cdef int H = a.shape[0]
    cdef int W = a.shape[1]
    cdef int M = a.shape[1] // 3
    cdef int V = i.shape[1]

    cdef VALUE_TYPE * ptr_a = < VALUE_TYPE * > < uintptr_t > a._ptr
    cdef VALUE_TYPE * ptr_b = < VALUE_TYPE * > < uintptr_t > b._ptr
    cdef VALUE_TYPE * ptr_c = < VALUE_TYPE * > < uintptr_t > c._ptr
    cdef VALUE_TYPE * ptr_d = < VALUE_TYPE * > < uintptr_t > d._ptr
    cdef VALUE_TYPE * ptr_e = < VALUE_TYPE * > < uintptr_t > e._ptr
    cdef VALUE_TYPE * ptr_f = < VALUE_TYPE * > < uintptr_t > f._ptr
    cdef VALUE_TYPE * ptr_g = < VALUE_TYPE * > < uintptr_t > g._ptr
    cdef VALUE_TYPE * ptr_h = < VALUE_TYPE * > < uintptr_t > h._ptr
    cdef VALUE_TYPE * ptr_i = < VALUE_TYPE * > < uintptr_t > i._ptr
    thrust_backward_gru(H, W, M, V, ptr_a, ptr_b, ptr_c, ptr_d, ptr_e, ptr_f, ptr_g, ptr_h, ptr_i)


def cubinarize(gpu_value1, th, gpu_value2):
    cdef int N = gpu_value1.size
    cdef VALUE_TYPE * gpu_ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * gpu_ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    cdef VALUE_TYPE threathold = th
    cuda_base.check_heap_device(gpu_value1, gpu_value2)
    thrust_binarize(gpu_ptr1, threathold, N, gpu_ptr2)


def cuembedding_forward(gpu_value1, weight, gpu_value2):
    cdef int N = gpu_value1.shape[0]
    cdef int K = weight.shape[0]
    cdef int M = weight.shape[1]
    cdef VALUE_TYPE * gpu_ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    cdef VALUE_TYPE * gpu_ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    cdef VALUE_TYPE * weight_ptr = <VALUE_TYPE * > < uintptr_t > weight._ptr
    cuda_base.check_heap_device(gpu_value1, gpu_value2, weight)
    thrust_embedding_forward(N, K, M, gpu_ptr1, weight_ptr, gpu_ptr2)


def cuembedding_backward(gpu_index, gpu_dy, gpu_dx):
    cdef int N = gpu_index.shape[0]
    cdef int K = gpu_dx.shape[0]
    cdef int M = gpu_dx.shape[1]
    cdef VALUE_TYPE * index_ptr = <VALUE_TYPE * > < uintptr_t > gpu_index._ptr
    cdef VALUE_TYPE * dy_ptr = <VALUE_TYPE * > < uintptr_t > gpu_dy._ptr
    cdef VALUE_TYPE * dx_ptr = <VALUE_TYPE * > < uintptr_t > gpu_dx._ptr
    cuda_base.check_heap_device(gpu_dy, gpu_index, gpu_dx)
    thrust_embedding_backward(N, K, M, index_ptr, dy_ptr, dx_ptr)


def cuconcat(gpu_values, gpu_value2, axis):
    for i in range(len(gpu_values[:-1])):
        cuda_base.check_heap_device(gpu_values[i], gpu_values[i + 1], gpu_value2)

    buffer_size = np.sum([val.nbytes for val in gpu_values])
    if gpu_value2.nbytes < buffer_size:
        raise ValueError("Insufficient destination buffer size")

    cdef size_t rec_size = 0
    for gpu_value in gpu_values:
        if (not gpu_value.shape):
            raise ValueError("zero-dimensional arrays cannot be concatenated")
        rec_size += functools.reduce(operator.__mul__, gpu_value.shape[axis:], 1)

    cdef size_t size = 0
    cdef concated_size
    cdef VALUE_TYPE * ptr1
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value2._ptr
    for gpu_value in gpu_values:
        s1 = gpu_value.shape[:axis] + gpu_value.shape[axis + 1:]
        concated_size = <int > functools.reduce(operator.__mul__, gpu_value.shape[axis:], 1)
        ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value._ptr
        thrust_copy_memory_stride(ptr2 + size, ptr1, gpu_value.size, rec_size, concated_size)
        size += <int > concated_size


ctypedef object(*REDUCE_FUNC)(
    size_t max_grids, size_t num_threads,
    VALUE_TYPE * src, size_t src_size,
    object result_shape, size_t result_size,
    size_t src_per_result,
    size_t sequence_stride,
    size_t num_axis,
    reduce_shape_infos * reductions_infos,
    reduce_shape_infos * seqs_infos,
    object args)


import collections


def _del_items(src, indexes):
    ret = list(src)
    for i in reversed(indexes):
        del ret[i]
    return ret


def _calc_index(reductions, kept_shapes_size, n):
    ret = 0
    if kept_shapes_size:
        ret = n % kept_shapes_size

    for info in reductions:
        v = n
        if info.group_size:
            v = v % info.group_size
        v = v // info.out_size
        ret += v * info.in_size

    return ret


cdef _reduce_array(max_grids, num_threads, gpu_value1, axis, keepdims, REDUCE_FUNC func, args):
    assert num_threads < 600

    if not gpu_value1.shape:
        return gpu_value1

    if isinstance(axis, int):
        axis = [axis]
    elif not axis:
        axis = list(range(len(gpu_value1.shape)))

    axis = list(sorted(set(axis)))

    if (max(axis) >= len(gpu_value1.shape)) or (min(axis) < 0):
        raise ValueError('Invalid axis: %s' % (axis,))

    if len(axis) == len(gpu_value1.shape):
        reduce_axis = [0]
        src_shape = (gpu_value1.size,)
        src_size = gpu_value1.size

        result_shape = ()
        result_size = 1
    else:
        reduce_axis = axis
        src_shape = gpu_value1.shape
        src_size = gpu_value1.size

        result_shape = _del_items(src_shape, reduce_axis)
        result_size = functools.reduce(operator.__mul__, result_shape, 1)

    if len(reduce_axis) >= RENOM_CUDA_MAX_AXIS:
        raise ValueError("Number of axis should be less than %d" % RENOM_CUDA_MAX_AXIS)

    kept_shapes = src_shape[reduce_axis[-1] + 1:]
    kept_shapes_size = functools.reduce(operator.__mul__, kept_shapes, 1)

    src_per_result = src_size // result_size
    sequence_per_result = src_shape[reduce_axis[0]]
    sequence_stride = kept_shapes_size
    src_per_sequence = src_per_result // sequence_per_result

    max_threads_per_result = min(src_per_result, num_threads)
    preferred_result_per_block = num_threads // max_threads_per_result

    num_blocks = min((result_size - 1) // preferred_result_per_block + 1, max_grids)

    cdef reduce_shape_infos reduction_infos
    group_size = 0
    f = 0

    for n, i in enumerate(reduce_axis):
        in_shape = src_shape[i:]
        in_size = functools.reduce(operator.__mul__, in_shape, 1)
        out_shape = _del_items(src_shape[i + 1:], [p - i - 1 for p in reduce_axis[n + 1:]])
        out_size = functools.reduce(operator.__mul__, out_shape, 1)

        reduction_infos.in_size[n] = in_size
        reduction_infos.out_size[n] = out_size
        reduction_infos.group_size[n] = group_size

        group_size = out_size

    cdef reduce_shape_infos seq_infos

    group_size = 0
    f = 0
    for n, i in enumerate(reduce_axis):
        in_shape = src_shape[i + 1:]
        in_size = functools.reduce(operator.__mul__, in_shape, 1)
        out_shape = [src_shape[p] for p in reduce_axis[n + 1:]]
        out_size = functools.reduce(operator.__mul__, out_shape, 1)

        seq_infos.in_size[n] = in_size
        seq_infos.out_size[n] = out_size
        seq_infos.group_size[n] = group_size

        group_size = out_size

    if not keepdims:
        ret_shape = result_shape
    else:
        ret_shape = list(gpu_value1.shape)
        for s in axis:
            ret_shape[s] = 1

    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr

    return func(num_blocks, num_threads, ptr1, src_size, ret_shape, result_size, src_per_result, sequence_stride,
                len(reduce_axis), & reduction_infos, & seq_infos, args)


cdef _cusum(size_t max_grids, size_t num_threads,
            VALUE_TYPE * src, size_t src_size,
            object result_shape, size_t result_size,
            size_t src_per_result,
            size_t sequence_stride,
            size_t num_axis,
            reduce_shape_infos * reductions_infos,
            reduce_shape_infos * seqs_infos,
            object args):

    result = renom.core.GPUValue(shape=result_shape)
    cdef VALUE_TYPE * ptr = <VALUE_TYPE * > < uintptr_t > result._ptr

    thrust_reduce_sum(max_grids, num_threads,
                      src, src_size,
                      ptr, result_size,
                      src_per_result,
                      sequence_stride,
                      num_axis,
                      reductions_infos,
                      seqs_infos)

    return result


def cusum(gpu_value1, axis=None, keepdims=False, max_grids=65536, num_threads=512):
    return _reduce_array(max_grids, num_threads, gpu_value1, axis, keepdims, _cusum, None)


cdef _cu_reduce_min(size_t max_grids, size_t num_threads,
                    VALUE_TYPE * src, size_t src_size,
                    object result_shape, size_t result_size,
                    size_t src_per_result,
                    size_t sequence_stride,
                    size_t num_axis,
                    reduce_shape_infos * reductions_infos,
                    reduce_shape_infos * seqs_infos,
                    object args):

    result = renom.core.GPUValue(shape=result_shape)
    cdef VALUE_TYPE * ptr = <VALUE_TYPE * > < uintptr_t > result._ptr

    thrust_reduce_min(max_grids, num_threads,
                      src, src_size,
                      ptr, result_size,
                      src_per_result,
                      sequence_stride,
                      num_axis,
                      reductions_infos,
                      seqs_infos)

    return result


def cu_reduce_min(gpu_value1, axis=None, keepdims=False, max_grids=65536, num_threads=512):
    return _reduce_array(max_grids, num_threads, gpu_value1, axis, keepdims, _cu_reduce_min, None)


cdef _cu_reduce_max(size_t max_grids, size_t num_threads,
                    VALUE_TYPE * src, size_t src_size,
                    object result_shape, size_t result_size,
                    size_t src_per_result,
                    size_t sequence_stride,
                    size_t num_axis,
                    reduce_shape_infos * reductions_infos,
                    reduce_shape_infos * seqs_infos,
                    object args):

    result = renom.core.GPUValue(shape=result_shape)
    cdef VALUE_TYPE * ptr = <VALUE_TYPE * > < uintptr_t > result._ptr

    thrust_reduce_max(max_grids, num_threads,
                      src, src_size,
                      ptr, result_size,
                      src_per_result,
                      sequence_stride,
                      num_axis,
                      reductions_infos,
                      seqs_infos)

    return result


def cu_reduce_max(gpu_value1, axis=None, keepdims=False, max_grids=65536, num_threads=512):
    return _reduce_array(max_grids, num_threads, gpu_value1, axis, keepdims, _cu_reduce_max, None)


cdef _cu_reduce_argmin(size_t max_grids, size_t num_threads,
                       VALUE_TYPE * src, size_t src_size,
                       object result_shape, size_t result_size,
                       size_t src_per_result,
                       size_t sequence_stride,
                       size_t num_axis,
                       reduce_shape_infos * reductions_infos,
                       reduce_shape_infos * seqs_infos,
                       object args):

    result = renom.core.GPUValue(shape=result_shape, dtype='int64')
    cdef size_t * ptr = <size_t * > < uintptr_t > result._ptr

    cdef size_t mod, div
    mod, div = args

    thrust_reduce_argmin(max_grids, num_threads,
                         src, src_size,
                         ptr, result_size,
                         src_per_result,
                         sequence_stride,
                         num_axis,
                         reductions_infos,
                         seqs_infos,
                         mod, div)

    return result


def cu_reduce_argmin(gpu_value1, axis=None, max_grids=65536, num_threads=512):
    if axis is not None:
        if not isinstance(axis, int) or axis >= len(gpu_value1.shape):
            raise ValueError("Invalid axis")

        mod = functools.reduce(operator.__mul__, gpu_value1.shape[axis:], 1)
        div = functools.reduce(operator.__mul__, gpu_value1.shape[axis + 1:], 1)

    else:
        mod = functools.reduce(operator.__mul__, gpu_value1.shape, 1)
        div = 1

    keepdims = False
    return _reduce_array(max_grids, num_threads, gpu_value1, axis, keepdims, _cu_reduce_argmin, (mod, div))


cdef _cu_reduce_argmax(size_t max_grids, size_t num_threads,
                       VALUE_TYPE * src, size_t src_size,
                       object result_shape, size_t result_size,
                       size_t src_per_result,
                       size_t sequence_stride,
                       size_t num_axis,
                       reduce_shape_infos * reductions_infos,
                       reduce_shape_infos * seqs_infos,
                       object args):

    result = renom.core.GPUValue(shape=result_shape, dtype='int64')
    cdef size_t * ptr = <size_t * > < uintptr_t > result._ptr

    cdef size_t mod, div
    mod, div = args

    thrust_reduce_argmax(max_grids, num_threads,
                         src, src_size,
                         ptr, result_size,
                         src_per_result,
                         sequence_stride,
                         num_axis,
                         reductions_infos,
                         seqs_infos,
                         mod, div)

    return result


def cu_reduce_argmax(gpu_value1, axis=None, max_grids=65536, num_threads=512):
    if axis is not None:
        if not isinstance(axis, int) or axis >= len(gpu_value1.shape):
            raise ValueError("Invalid axis")

        mod = functools.reduce(operator.__mul__, gpu_value1.shape[axis:], 1)
        div = functools.reduce(operator.__mul__, gpu_value1.shape[axis + 1:], 1)

    else:
        mod = functools.reduce(operator.__mul__, gpu_value1.shape, 1)
        div = 1

    keepdims = False

    return _reduce_array(max_grids, num_threads, gpu_value1, axis, keepdims, _cu_reduce_argmax, (mod, div))


def cu_add_bias(bias, gpu_value):
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > bias._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value._ptr
    cdef int size = <int > gpu_value.size
    cdef int wh
    if len(gpu_value.shape) < 4:
        wh = <int > (gpu_value.shape[2])
    elif len(gpu_value.shape) < 5:
        wh = <int > (gpu_value.shape[2] * gpu_value.shape[3])
    elif len(gpu_value.shape) is 5:
        wh = <int > (gpu_value.shape[2] * gpu_value.shape[3] * gpu_value.shape[4])
    else:
        assert False, "cu_add_bias currently supports only 2d or 3d biases"
    cdef int n = <int > gpu_value.shape[0]
    thrust_add_bias(size, n, wh, ptr1, ptr2)


def cu_get_fg_ary_forward(ary, fg_ary):
    N = ary.shape[0] * ary.shape[1] * ary.shape[2] * ary.shape[3] * ary.shape[4]
    M = ary.shape[3] * ary.shape[4]
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > ary._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > fg_ary._ptr
    thrust_get_fg_ary_forward(N, M, ptr1, ptr2)


def cu_get_fg_ary_backward(du, zero):
    N = zero.shape[0] * zero.shape[1] * zero.shape[2] * zero.shape[3] * zero.shape[4]
    M = du.shape[3] * du.shape[4]
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > du._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > zero._ptr
    thrust_get_fg_ary_forward(N, M, ptr1, ptr2)


def cu_get_ith_ary_forward(ary, ith_ary, i):
    N = ary.size
    M = ary.size / ary.shape[0]
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > ary._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > ith_ary._ptr
    thrust_get_ith_ary_forward(N, M, i, ptr1, ptr2)


def cu_get_ith_ary_backward(du, zero, i):
    N = zero.size
    M = zero.size / zero.shape[0]
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > du._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > zero._ptr
    thrust_get_ith_ary_forward(N, M, i, ptr1, ptr2)


def cu_get_every_nth_ary(ary1, ary2, i, j):
    N = ary1.shape[0]
    M = ary1.shape[1]
    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > ary1._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > ary2._ptr
    thrust_get_nth_ary(N, M, i, j, ptr1, ptr2)


def cu_assign_pred_box(x, y, w, h, ary):
    N, M = ary.shape
    cdef VALUE_TYPE * ary_ptr = <VALUE_TYPE * > < uintptr_t > ary._ptr
    cdef VALUE_TYPE * x_ptr = <VALUE_TYPE * > < uintptr_t > x._ptr
    cdef VALUE_TYPE * y_ptr = <VALUE_TYPE * > < uintptr_t > y._ptr
    cdef VALUE_TYPE * h_ptr = <VALUE_TYPE * > < uintptr_t > h._ptr
    cdef VALUE_TYPE * w_ptr = <VALUE_TYPE * > < uintptr_t > w._ptr
    thrust_assign_pred_box(N, M, x_ptr, y_ptr, h_ptr, w_ptr, ary_ptr)


def cu_pred_ctr(arg, length, ctr, ary):
    N, M = ary.shape
    cdef VALUE_TYPE * arg_ptr = <VALUE_TYPE * > < uintptr_t > arg._ptr
    cdef VALUE_TYPE * length_ptr = <VALUE_TYPE * > < uintptr_t > length._ptr
    cdef VALUE_TYPE * ctr_ptr = <VALUE_TYPE * > < uintptr_t > ctr._ptr
    cdef VALUE_TYPE * ary_ptr = <VALUE_TYPE * > < uintptr_t > ary._ptr
    thrust_pred_ctr(N, M, arg_ptr, length_ptr, ctr_ptr, ary_ptr)


def cu_generate_anchors(shifts, base_size, ratios, scales, feat_stride, anchors):
    K, A, N = anchors.shape
    scale_size = scales.shape[0]
    ratio_size = ratios.shape[0]
    cdef VALUE_TYPE * shifts_ptr = <VALUE_TYPE * > < uintptr_t > shifts._ptr
    cdef VALUE_TYPE * ratios_ptr = <VALUE_TYPE * > < uintptr_t > ratios._ptr
    cdef VALUE_TYPE * scales_ptr = <VALUE_TYPE * > < uintptr_t > scales._ptr
    cdef VALUE_TYPE * anchors_ptr = <VALUE_TYPE * > < uintptr_t > anchors._ptr
    thrust_generate_anchors(A, K, N, shifts_ptr, ratios_ptr, scales_ptr,
                            ratio_size, scale_size, feat_stride, base_size, anchors_ptr)


def cu_get_ith_bbox(bbox, i, ary):
    N, M = bbox.shape
    cdef VALUE_TYPE * bbox_ptr = <VALUE_TYPE * > < uintptr_t > bbox._ptr
    cdef VALUE_TYPE * ary_ptr = <VALUE_TYPE * > < uintptr_t > ary._ptr
    thrust_get_ith_bbox(N, M, bbox_ptr, i, ary_ptr)


def cu_clip_roi(roi, start, end, step, min_v, max_v, ary):
    N, M = roi.shape
    cdef VALUE_TYPE * roi_ptr = <VALUE_TYPE * > < uintptr_t > roi._ptr
    cdef VALUE_TYPE * ary_ptr = <VALUE_TYPE * > < uintptr_t > ary._ptr
    thrust_clip_roi(N, M, roi_ptr, start, end, step, min_v, max_v, ary_ptr)


def cu_transpose(gpu_value1, axis):
    # [np.prod(gpu_value1.shape[i + 1:], dtype='int') for i in range(len(gpu_value1.shape))]
    strides = calc_strides(gpu_value1.shape)

    if not axis:
        axis = tuple(reversed(range(len(gpu_value1.shape))))

    if len(axis) >= 16:
        raise ValueError('Invalid axis: %s' % (axis,))

    new_shape = [gpu_value1.shape[i] for i in axis]

    cdef size_t src_strides[16]
    for i, s in enumerate(axis):
        src_strides[i] = strides[s]

    cdef size_t new_strides[16]

    s = calc_strides(new_shape)
    for i in range(len(s)):
        new_strides[i] = s[i]

    cdef VALUE_TYPE * ptr = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr
    size = calc_int_prod(gpu_value1.shape)

    result = renom.core.GPUValue(shape=new_shape)
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > result._ptr

    thrust_transpose(size,
                     len(new_shape),
                     ptr, src_strides,
                     ptr2, new_strides)

    return result


cdef _build_slice_infos(getitem_slice_infos * infos, slices):
    if len(slices) >= RENOM_CUDA_MAX_AXIS:
        raise ValueError("Number of axis should be less than %d" % RENOM_CUDA_MAX_AXIS)

    infos.shape_len = len(slices)
    for i, (start, stop, step, adv_indexes, stride, dest_stride) in enumerate(slices):
        infos.slice_info[i].start = start
        infos.slice_info[i].stop = stop
        infos.slice_info[i].step = step
        if adv_indexes:
            infos.slice_info[i].adv_indexes_len = adv_indexes.size
            infos.slice_info[i].adv_indexes = <long long * > < uintptr_t > adv_indexes._ptr
        else:
            infos.slice_info[i].adv_indexes_len = 0
            infos.slice_info[i].adv_indexes = NULL

        infos.slice_info[i].stride = stride
        infos.slice_info[i].dest_stride = dest_stride


def cu_get_item(gpu_value1, size, dest_size, slices):

    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr

    result = renom.core.GPUValue(shape=(dest_size,))
    cdef VALUE_TYPE * ptr_result = <VALUE_TYPE * > < uintptr_t > result._ptr

    cdef getitem_slice_infos infos
    _build_slice_infos( & infos, slices)

    cdef getitem_slice_info * info

    thrust_getitem(ptr1, ptr_result, dest_size, & infos)

    return result


def cu_set_item(value, valuesize, gpu_value1, slices, strides, broadcasted_strides):
    if not isinstance(value, renom.core.GPUValue):
        if isinstance(value, renom.core.Node):
            value = value.get_gpu()
        elif isinstance(value, np.ndarray):
            value = renom.core.GPUValue(array=value)
        else:
            value = renom.core.GPUValue(array=np.array(value))

    if value.dtype.name != gpu_value1.dtype.name:
        raise ValueError()

    cdef VALUE_TYPE * ptr1 = <VALUE_TYPE * > < uintptr_t > value._ptr
    cdef VALUE_TYPE * ptr2 = <VALUE_TYPE * > < uintptr_t > gpu_value1._ptr

    cdef getitem_slice_infos infos
    _build_slice_infos( & infos, slices)

    infos.stride_size = len(strides)
    for i, (s, b) in enumerate(zip(strides, broadcasted_strides)):
        infos.strides[i] = s
        infos.broadcasted_strides[i] = b

    thrust_setitem(ptr1, valuesize, ptr2, & infos)


def cu_optimizer_sgd(learning_rate, momentum, dy, previous_dy, new_dy):
    Elem = 1
    for v in dy.shape:
        Elem *= v
    cdef int Elems = Elem
    cdef VALUE_TYPE lr = learning_rate
    cdef VALUE_TYPE mo = momentum
    cdef VALUE_TYPE * ptr_dy = <VALUE_TYPE * > < uintptr_t > dy._ptr
    cdef VALUE_TYPE * ptr_pdy = <VALUE_TYPE * > < uintptr_t > previous_dy._ptr
    cdef VALUE_TYPE * ptr_ndy = <VALUE_TYPE * > < uintptr_t > new_dy._ptr
    thrust_optimizer_sgd(Elems, lr, ptr_dy, mo, ptr_pdy, ptr_ndy)


def cu_optimizer_adagrad(learning_rate, epsilon, dy, previous_dy, new_dy, r):
    Elem = 1
    for v in dy.shape:
        Elem *= v
    cdef int Elems = Elem
    cdef VALUE_TYPE lr = learning_rate
    cdef VALUE_TYPE eps = epsilon
    cdef VALUE_TYPE * ptr_dy = <VALUE_TYPE * > < uintptr_t > dy._ptr
    cdef VALUE_TYPE * ptr_pdy = <VALUE_TYPE * > < uintptr_t > previous_dy._ptr
    cdef VALUE_TYPE * ptr_ndy = <VALUE_TYPE * > < uintptr_t > new_dy._ptr
    cdef VALUE_TYPE * ptr_r = <VALUE_TYPE * > < uintptr_t > r._ptr
    thrust_optimizer_adagrad(Elems, lr, ptr_dy, eps, ptr_pdy, ptr_ndy, ptr_r)


def cu_optimizer_rmsprop(learning_rate, epsilon, gamma, eta, dy, new_dy, r):
    Elem = 1
    for v in dy.shape:
        Elem *= v
    cdef int Elems = Elem
    cdef VALUE_TYPE lr = learning_rate
    cdef VALUE_TYPE eps = epsilon
    cdef VALUE_TYPE g = gamma
    cdef VALUE_TYPE e = eta
    cdef VALUE_TYPE * ptr_dy = <VALUE_TYPE * > < uintptr_t > dy._ptr
    cdef VALUE_TYPE * ptr_ndy = <VALUE_TYPE * > < uintptr_t > new_dy._ptr
    cdef VALUE_TYPE * ptr_r = <VALUE_TYPE * > < uintptr_t > r._ptr
    thrust_optimizer_rmsprop(Elems, lr, ptr_dy, eps, g, e, ptr_ndy, ptr_r)


def cu_optimizer_adam(learning_rate, epsilon, gamma, gamma_orig, beta, beta_orig, minimum, toflug, u, r, dy, new_dy):
    Elem = 1
    for v in dy.shape:
        Elem *= v
    cdef int Elems = Elem
    cdef VALUE_TYPE lr = learning_rate
    cdef VALUE_TYPE eps = epsilon
    cdef VALUE_TYPE g = gamma
    cdef VALUE_TYPE go = gamma_orig
    cdef VALUE_TYPE b = beta
    cdef VALUE_TYPE bo = beta_orig
    cdef VALUE_TYPE min = minimum
    cdef bool flug = toflug
    cdef VALUE_TYPE * ptr_u = <VALUE_TYPE * > < uintptr_t > u._ptr
    cdef VALUE_TYPE * ptr_r = <VALUE_TYPE * > < uintptr_t > r._ptr
    cdef VALUE_TYPE * ptr_dy = <VALUE_TYPE * > < uintptr_t > dy._ptr
    cdef VALUE_TYPE * ptr_ndy = <VALUE_TYPE * > < uintptr_t > new_dy._ptr
    thrust_optimizer_adam(Elems, lr, ptr_dy, eps, g, go, b,
                          bo, min, flug, ptr_u, ptr_r, ptr_ndy)


def cu_clip(array, minimum, maximum):
    cdef int Elem = 1
    for v in array.shape:
        Elem *= <int > v
    cdef VALUE_TYPE max = <VALUE_TYPE > maximum
    cdef VALUE_TYPE min = <VALUE_TYPE > minimum
    cdef VALUE_TYPE * ptr_arr = <VALUE_TYPE * > < uintptr_t > array._ptr
    thrust_clip(Elem, ptr_arr, maximum, minimum)


def cu_optimizer_adadelta(decay_rate, epsilon, previous_squared_gradient, previous_squared_delta, dy, new_dy):
    cdef int Elem = 1
    for v in dy.shape:
        Elem *= v
    cdef VALUE_TYPE dr = decay_rate
    cdef VALUE_TYPE eps = epsilon
    cdef VALUE_TYPE * ptr_psg = <VALUE_TYPE * > < uintptr_t > previous_squared_gradient._ptr
    cdef VALUE_TYPE * ptr_psx = <VALUE_TYPE * > < uintptr_t > previous_squared_delta._ptr
    cdef VALUE_TYPE * ptr_dy = <VALUE_TYPE * > < uintptr_t > dy._ptr
    cdef VALUE_TYPE * ptr_ndy = <VALUE_TYPE * > < uintptr_t > new_dy._ptr
    thrust_optimizer_adadelta(Elem, dr, eps, ptr_psg, ptr_psx, ptr_dy, ptr_ndy)


def cu_optimizer_adamax(alpha, epsilon, beta1, beta2, moment1, moment2, dy, new_dy):
    cdef int Elem = 1
    for v in dy.shape:
        Elem *= v
    cdef VALUE_TYPE alp = alpha
    cdef VALUE_TYPE eps = epsilon
    cdef VALUE_TYPE b_1 = beta1[0]
    cdef VALUE_TYPE rb_1 = beta1[1]
    cdef VALUE_TYPE b_2 = beta2[0]
    cdef VALUE_TYPE rb_2 = beta2[1]
    cdef VALUE_TYPE * ptr_mom1 = <VALUE_TYPE * > < uintptr_t > moment1._ptr
    cdef VALUE_TYPE * ptr_mom2 = <VALUE_TYPE * > < uintptr_t > moment2._ptr
    cdef VALUE_TYPE * ptr_dy = <VALUE_TYPE * > < uintptr_t > dy._ptr
    cdef VALUE_TYPE * ptr_ndy = <VALUE_TYPE * > < uintptr_t > new_dy._ptr
    thrust_optimizer_adamax(Elem, alp, eps, b_1, rb_1, b_2, rb_2,
                            ptr_mom1, ptr_mom2, ptr_dy, ptr_ndy)
