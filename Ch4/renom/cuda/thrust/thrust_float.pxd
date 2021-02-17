
cdef extern from "thrust_funcs_float.h":
    ctypedef float VALUE_TYPE

include "thrust_func_defs.pxi"

cdef extern from * namespace "renom":
    cdef void set_stream_float(cudaStream_t stream)
