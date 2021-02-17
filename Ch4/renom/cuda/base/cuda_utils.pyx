import contextlib
from cpython cimport Py_buffer, PyObject_GetBuffer, PyBuffer_Release

cdef class _VoidPtr:
    def __init__(self, object):
        self.value = object
        PyObject_GetBuffer(object, &(self.buf), 0)
        self.ptr = self.buf.buf

    def __dealloc__(self):
        PyBuffer_Release(&(self.buf))

