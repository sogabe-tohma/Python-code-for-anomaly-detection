
cdef extern from "thrust_funcs_double.h":
    ctypedef double VALUE_TYPE

include "thrust_func_defs.pxi"

cdef extern from * namespace "renom":
    cdef void set_stream_double(cudaStream_t stream)
