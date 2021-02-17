cimport thrust_double as renom_thrust

def cu_set_stream(stream):
    cdef cudaStream_t _stream = <cudaStream_t><uintptr_t> stream
    set_stream_double(_stream)

include "thrust_funcs.pxi"
