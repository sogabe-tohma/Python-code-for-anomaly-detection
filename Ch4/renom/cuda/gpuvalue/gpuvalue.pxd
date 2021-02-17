# distutils: language=c++

from renom.cuda.cublas cimport cublas
from renom.cuda.base cimport cuda_base

cdef class _AdvIndex:
    cdef public object org_index
    cdef public object index
    cdef public object shape


cpdef _parse_index(arr, indexes)
cpdef build_shapes(arr, indexes)
cpdef _build_broadcast_mask(left, right)


cdef class GPUValue:
    cdef object __weakref__

    cdef public cuda_base.GPUHeap _ptr
    cdef public tuple shape
    cdef public object dtype
    cdef public size_t itemsize
    cdef public size_t size
    cdef public size_t nbytes
    cdef public int device_id

    cpdef alloc(self)
    cpdef _free(self)
    cpdef get_gpu(self)
    cpdef copy(self)
    cpdef empty_like_me(self)
    cpdef zeros_like_me(self)
    cpdef ones_like_me(self)
    cpdef new_array(self)
    cpdef to_cpu(self, value)
    cpdef to_gpu(self, value)
    cpdef copy_from(self, other)
    cpdef transpose(self, axis)
    cpdef split(self, indices_or_sections, axis=*)
    cpdef hsplit(self, indices_or_sections)
