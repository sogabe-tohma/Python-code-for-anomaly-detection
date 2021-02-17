from cpython cimport Py_buffer

cdef class _VoidPtr:
    cdef object value
    cdef Py_buffer buf
    cdef void *ptr
