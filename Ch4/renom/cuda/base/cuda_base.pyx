from __future__ import print_function
import atexit
import sys
import traceback
import contextlib
import bisect
import threading
cimport cuda_base

from libc.stdio cimport printf
cimport numpy as np
import numpy as pnp
cimport cython
from numbers import Number
from libc.stdlib cimport malloc, free
from libc.stdint cimport uintptr_t, intptr_t
from libc.string cimport memcpy
from cuda_utils cimport _VoidPtr
from renom.config import precision
import collections
import renom.cuda

# Indicate Python started shutdown process
cdef int _python_shutdown = 0

@atexit.register
def on_exit():
    _python_shutdown = 1


@contextlib.contextmanager
def use_device(device_id):
    active = renom.cuda.is_cuda_active()
    cdef int cur

    if active:
        cur = cuGetDevice()
        cuSetDevice(device_id)  # switch dedice

    try:
        yield
    finally:
        if active:
            cuSetDevice(cur)   # restore device

def cuMalloc(uintptr_t nbytes):
    cdef void * p
    runtime_check(cudaMalloc( & p, nbytes))
    return < uintptr_t > p


cpdef cuMemset(uintptr_t ptr, int value, size_t size):
    p = <void * >ptr
    runtime_check(cudaMemset(p, value, size))
    return

# Global C-defined variables cannot be accessed from Python
cdef void ** ptrs # Keeps an array of pointers for storing pinned memory spaces
cdef cudaEvent_t * events # Similar to above with regards to events
# TODO: Make pin_size always word-sized for increased performance in memcpy
cdef size_t pin_size = 0 # How much space is designed for each pinned memory location
cdef int using = 0 # Designates which of the allocated spaces is currently being used
cdef int numPointers = 2 # How many pinned memory spaces we use
asyncTransferContexts = 0
usePinned = False

@contextlib.contextmanager
def asyncBehaviour():
  global usePinned, asyncTransferContexts
  usePinned = True
  asyncTransferContexts += 1
  yield
  asyncTransferContexts -= 1
  if asyncTransferContexts is 0:
    usePinned = False

# TODO: Create a list of pointers received in pinNumpy and make sure that each
# pointer is consumed only once! Any subsequent calls with the same address pointer value
# should be re-pinned again!


# This function initiates the pinned memory space based on a sample batch
# This should be called before trying to pin memory
def initPinnedMemory(np.ndarray batch_arr):
    global pin_size, events, ptrs, numPointers
    cdef int elems
    elems = getArrayElements(batch_arr)
    cdef size_t new_size = <size_t> batch_arr.descr.itemsize*elems
    if not (pin_size is new_size):
      if new_size is 0 and pin_size is 0:
        freePinnedMemory()
      elif new_size > 0:
        pin_size = new_size
        ptrs = <void**> malloc(sizeof(void*)*numPointers)
        events = <cudaEvent_t*> malloc(sizeof(cudaEvent_t)*numPointers)
        for i in range(numPointers):
          runtime_check(cudaMallocHost(&(ptrs[i]), pin_size))
          runtime_check(cudaEventCreate(&(events[i])))




# Pointless right now, as the user closing the program
# will always free up the allocated memory anyway
def freePinnedMemory():
    global ptrs, pin_size, numPointers
    if pin_size is 0:
      return
    cdef int i
    for i in range(numPointers):
      cudaFreeHost(ptrs[i])
    free(ptrs)
    free(events)
    pin_size = 0

cdef getArrayElements(np.ndarray arr):
  cdef int i, elems
  elems = 1
  for i in range(arr.ndim):
    elems *= arr.shape[i]
  return elems

cdef pinPointer(void* cpu_ptr, size_t nbytes):
  assert nbytes <= pin_size
  cudaEventSynchronize(events[using])
  memcpy(ptrs[using], cpu_ptr, nbytes)
# This function will accept a numpy array and move its data to the spaces
# previously allocated using initPinnedMemory.
def pinNumpy(np.ndarray arr):
    global ptrs, using, events, pin_size
    cdef int elems
    elems = getArrayElements(arr)
    cdef size_t arr_size = <size_t> arr.descr.itemsize*elems
    assert arr_size <= pin_size, "Attempting to insert memory larger than what was made available through initPinnedMemory.\n(Allocated,Requested)({:d},{:d})".format(pin_size,arr.descr.itemsize*elems)
    cdef void * vptr = <void*> arr.data
    cudaEventSynchronize(events[using])
    #memcpy(ptrs[using], vptr, pin_size)
    memcpy(ptrs[using], vptr, arr_size)
    #using = (using + 1) % numPointers
    return

'''
Creates a stream
The name is optional, if not given a default name will be chosen

A cudaStream_t type is a pointer to a CUstream_st struct
The function return value is the integer converted value of this pointer
To reuse this stream as a C-defined cudaStream_t variable, simply cast the
returned integer value back to cudaStream_t
'''
def cuCreateStream(name = None):
    cdef cudaStream_t stream
    cdef char* cname
    #runtime_check(cudaStreamCreateWithFlags( & stream, cudaStreamNonBlocking))
    runtime_check(cudaStreamCreate( & stream ))
    if name is not None:
      py_byte_string = name.encode("UTF-8")
      cname = py_byte_string
      nvtxNameCudaStreamA(stream, cname)
    return < uintptr_t > stream

cdef cudaStream_t mainstream = <cudaStream_t><uintptr_t> 0
cdef cudaStream_t cudnnstream = <cudaStream_t><uintptr_t> 0


def setMainStream(stream):
    global mainstream
    mainstream = <cudaStream_t><uintptr_t> stream

def setCudnnStream(stream):
  global cudnnstream
  cudnnstream = <cudaStream_t><uintptr_t> stream

def insertEvent(GPUHeap heap):
  global mainstream
  runtime_check(cudaEventRecord(heap.event, mainstream))

def heapReady(GPUHeap heap):
  ret = cudaEventQuery(heap.event)
  if ret == cudaSuccess:
    return True
  else:
    return False

def cuDestroyStream(uintptr_t stream):
    runtime_check(cudaStreamDestroy(<cudaStream_t> stream))

def cuResetDevice():
  runtime_check(cudaDeviceReset())

def cuGetMemInfo():
    cdef size_t free, total
    cudaMemGetInfo(&free, &total)
    return <long> free, <long> (total-free), <long> total # free, used, total


def cuSetDevice(int dev):
    runtime_check(cudaSetDevice(dev))


cpdef int cuGetDevice():
    cdef int dev
    runtime_check(cudaGetDevice(&dev))
    return dev

cpdef cuDeviceSynchronize():
    runtime_check(cudaDeviceSynchronize())


cpdef cuCreateCtx(device=0):
    cdef CUcontext ctx
    driver_check(cuCtxCreate( & ctx, 0, device))
    return int(ctx)


cpdef cuGetDeviceCxt():
    cdef CUdevice device
    driver_check(cuCtxGetDevice( & device))
    return int(device)


cpdef cuGetDeviceCount():
    cdef int count
    runtime_check(cudaGetDeviceCount( & count))
    return int(count)


cpdef cuGetDeviceProperty(device):
    cdef cudaDeviceProp property
    runtime_check(cudaGetDeviceProperties( & property, device))
    property_dict = {
        "name": property.name,
        "totalGlobalMem": property.totalGlobalMem,
        "sharedMemPerBlock": property.sharedMemPerBlock,
        "regsPerBlock": property.regsPerBlock,
        "warpSize": property.warpSize,
        "memPitch": property.memPitch,
        "maxThreadsPerBlock": property.maxThreadsPerBlock,
        "maxThreadsDim": property.maxThreadsDim,
        "maxGridSize": property.maxGridSize,
        "totalConstMem": property.totalConstMem,
        "major": property.major,
        "minor": property.minor,
        "clockRate": property.clockRate,
        "textureAlignment": property.textureAlignment,
        "deviceOverlap": property.deviceOverlap,
        "multiProcessorCount": property.multiProcessorCount,
        "kernelExecTimeoutEnabled": property.kernelExecTimeoutEnabled,
        "computeMode": property.computeMode,
        "concurrentKernels": property.concurrentKernels,
        "ECCEnabled": property.ECCEnabled,
        "pciBusID": property.pciBusID,
        "pciDeviceID": property.pciDeviceID,
        "tccDriver": property.tccDriver,
    }

    return property_dict


cpdef cuFree(uintptr_t ptr):
    p = <void * >ptr
    runtime_check(cudaFree(p))
    return

# cuda runtime check
cpdef runtime_check(error):
    if error != cudaSuccess:
        error_msg = cudaGetErrorString(error)
        raise Exception("CUDA Error: #{}|||{}".format(error,error_msg))
    return

# cuda runtime check
cpdef driver_check(error):
    cdef char * string
    if error != 0:
        cuGetErrorString(error, < const char**> & string)
        error_msg = str(string)
        raise Exception(error_msg)
    return

# Memcpy
# TODO: in memcpy function, dest arguments MUST come first!

cdef void cuMemcpyH2D(void* cpu_ptr, uintptr_t gpu_ptr, int size):
    # cpu to gpu
    runtime_check(cudaMemcpy(<void *>gpu_ptr, cpu_ptr, size, cudaMemcpyHostToDevice))
    return

def queryDeviceProperties():
    cdef cudaDeviceProp props
    cdef int device = 0
    cudaGetDeviceProperties(&props,device)
    print("Device name is {}".format(
      props.name
    ))
    print("Device has compute capability {:d}.{:d}".format(
      props.major, props.minor
    ))
    print("Device has {:d} engines".format(
      props.asyncEngineCount
    ))


cdef cuMemcpyD2H(uintptr_t gpu_ptr, void *cpu_ptr, int size):
    # gpu to cpu
    runtime_check(cudaMemcpy(cpu_ptr, <void *>gpu_ptr, size, cudaMemcpyDeviceToHost))
    return

cdef cuMemcpyD2Hvar(uintptr_t gpu_ptr, void *cpu_ptr, int size, uintptr_t stream):
    # gpu to cpu
    runtime_check(cudaMemcpy(cpu_ptr, <void *>gpu_ptr, size, cudaMemcpyDeviceToHost))
    return


def cuMemcpyD2D(uintptr_t gpu_ptr1, uintptr_t gpu_ptr2, int size):
    # gpu to gpu
    runtime_check(cudaMemcpy(< void*>gpu_ptr2, < const void*>gpu_ptr1, size, cudaMemcpyDeviceToDevice))
    return

cdef void cuMemcpyH2DAsync(void* cpu_ptr, uintptr_t gpu_ptr, int size, uintptr_t stream):
    # cpu to gpu
    global ptrs, using, events, numPointers
    if pin_size > 0:
      cpu_ptr = ptrs[using]
    runtime_check(cudaMemcpyAsync(<void *>gpu_ptr, cpu_ptr, size, cudaMemcpyHostToDevice, <cudaStream_t> stream))
    if pin_size > 0:
      cudaEventRecord(events[using], <cudaStream_t> stream)
      cudaStreamWaitEvent(NULL, events[using], 0)
      using = (using + 1) % numPointers
    return

cdef printPtrSum():
  global ptrs, pin_size, using
  cdef float * ptr = <float*> ptrs[using]
  cdef int i, elems
  cdef float sum = 0
  elems = pin_size / sizeof(float)
  for i in range(elems-1):
    sum += ptr[i]
  print("Range sum is: {}".format(sum))

cdef void cuMemcpyH2Dvar(void* cpu_ptr, uintptr_t gpu_ptr, int size, uintptr_t stream):
  # cpu to gpu
  global ptrs, using, events, numPointers, pin_size, mainstream
  if pin_size > 0 and size <= pin_size and usePinned:
    runtime_check(cudaMemcpyAsync(<void *>gpu_ptr, ptrs[using], size, cudaMemcpyHostToDevice, <cudaStream_t> stream))
    runtime_check(cudaEventRecord(events[using], (<cudaStream_t> stream)))
    runtime_check(cudaStreamWaitEvent(mainstream, events[using], 0))
    using = (using + 1) % numPointers
  else:
    runtime_check(cudaMemcpy(<void *>gpu_ptr, cpu_ptr, size, cudaMemcpyHostToDevice))
  return


def cuMemcpyD2HAsync(uintptr_t gpu_ptr, np.ndarray[float, ndim=1, mode="c"] cpu_ptr, int size, int stream=0):
    # gpu to cpu
    runtime_check(cudaMemcpyAsync( & cpu_ptr[0], < const void*>gpu_ptr, size, cudaMemcpyDeviceToHost, < cudaStream_t > stream))
    return


cpdef cuMemcpyD2DAsync(uintptr_t gpu_ptr1, uintptr_t gpu_ptr2, int size, int stream=0):
    # gpu to gpu
    runtime_check(cudaMemcpyAsync( < void*>gpu_ptr2, < const void*>gpu_ptr1, size, cudaMemcpyDeviceToDevice, < cudaStream_t > stream))
    return

def check_heap_device(*heaps):
    devices = {h._ptr.device_id for h in heaps if isinstance(h, renom.core.GPUValue)}

    current = {cuGetDevice()}
    if devices != current:
        raise RuntimeError('Invalid device_id: %s currennt: %s' % (devices, current))


cdef class GPUHeap(object):
    def __init__(self, nbytes, ptr, device_id, stream=0):
        self.ptr = ptr
        self.nbytes = nbytes
        self.device_id = device_id
        # The GPUHeap sets its refcount to 0, as it does not personally know if it is
        # to be owned during creation. Refcount is instead managed in GPUValue.
        self.refcount = 0
        # The stream is decided by the allocator and given to all subsequently
        # constructed GPUHeaps. All Memcpy operations will occur on the same
        # stream.
        self._mystream = stream
        cudaEventCreate(&self.event)

    def __int__(self):
        return self.ptr

    def __dealloc__(self):
        # Python functions should be avoided as far as we can

        cdef int cur
        cdef cudaError_t err
        cdef const char *errstr;

        cudaGetDevice(&cur)

        try:
            err = cudaSetDevice(self.device_id)
            if err == cudaSuccess:
                err = cudaFree(<void * >self.ptr)

            if err != cudaSuccess:
                errstr = cudaGetErrorString(err)
                if _python_shutdown == 0:
                    s =  errstr.decode('utf-8', 'replace')
                    print("Error in GPUHeap.__dealloc__():", err, s, file=sys.stderr)
                else:
                    printf("Error in GPUHeap.__dealloc__(): %s\n", errstr)

        finally:
            cudaSetDevice(cur)


    cpdef memcpyH2D(self, cpu_ptr, size_t nbytes):
        # todo: this copy is not necessary
        # This pointer is already exposed with the Cython numpy implementation
        buf = cpu_ptr.ravel()
        cdef _VoidPtr ptr = _VoidPtr(buf)

        with renom.cuda.use_device(self.device_id):
            #cuMemcpyH2D(ptr.ptr, self.ptr, nbytes)
            cuMemcpyH2Dvar(ptr.ptr, self.ptr, nbytes, <uintptr_t> get_gpu_allocator()._memsync_stream)

    cpdef memcpyD2H(self, cpu_ptr, size_t nbytes):
        shape = cpu_ptr.shape
        cpu_ptr = pnp.reshape(cpu_ptr, -1)

        cdef _VoidPtr ptr = _VoidPtr(cpu_ptr)

        with renom.cuda.use_device(self.device_id):
            cuMemcpyD2H(self.ptr, ptr.ptr, nbytes)

        pnp.reshape(cpu_ptr, shape)

    cpdef memcpyD2D(self, gpu_ptr, size_t nbytes):
        assert self.device_id == gpu_ptr.device_id
        with renom.cuda.use_device(self.device_id):
            cuMemcpyD2D(self.ptr, gpu_ptr.ptr, nbytes)

    cpdef copy_from(self, other, size_t nbytes):
        cdef void *buf
        cdef int ret
        cdef uintptr_t src, dest

        assert nbytes <= self.nbytes
        assert nbytes <= other.nbytes

        n = min(self.nbytes, other.nbytes)
        if self.device_id == other.device_id:
            # self.memcpyD2D(other, n)
            other.memcpyD2D(self, n)
        else:
            runtime_check(cudaDeviceCanAccessPeer(&ret, self.device_id, other.device_id))
            if ret:
                src = other.ptr
                dest = self.ptr
                runtime_check(cudaMemcpyPeer(<void *>dest, self.device_id, <void*>src, other.device_id, nbytes))
            else:
                buf = malloc(n)
                if not buf:
                    raise MemoryError()
                try:
                    with renom.cuda.use_device(other.device_id):
                        cuMemcpyD2H(other.ptr, buf, n)

                    with renom.cuda.use_device(self.device_id):
                        cuMemcpyH2D(buf, self.ptr, n)

                finally:
                    free(buf)


cdef class GpuAllocator(object):

    def __init__(self):
        self._pool_lists = collections.defaultdict(list)
        # We create one stream for all the GPUHeaps to share
        self._memsync_stream = cuCreateStream("Memcpy Stream")
        self._rlock = threading.RLock()

    @property
    def pool_list(self):
        device = cuGetDevice()
        return self._pool_lists[device]

    cpdef GPUHeap malloc(self, size_t nbytes):
        cdef GPUHeap pool = self.getAvailablePool(nbytes)
        if pool is None:
            ptr = cuMalloc(nbytes)
            pool = GPUHeap(nbytes=nbytes, ptr=ptr, device_id=cuGetDevice())
        return pool


    cpdef free(self, GPUHeap pool):
        '''
        When a pool is to be freed, we first record the current status of the stream in which it was used,
        so as to make sure that it is not prematurely released for use by other GPUValues requesting a pool.
        '''
        global mainstream
        if _python_shutdown:
            return

        if not <uintptr_t> mainstream == 0:
            insertEvent(pool)

        if pool.nbytes:
            device_id = pool.device_id

            with self._rlock:
                self._pool_lists[device_id]
                index = bisect.bisect(self._pool_lists[device_id], (pool.nbytes,))
                self._pool_lists[device_id].insert(index, (pool.nbytes, pool))

    cpdef GPUHeap getAvailablePool(self, size_t size):
        pool = None
        '''
        We will be looking through the currently available pools and we demand that they
        big enough to fit all our requested data, but we allow for pools that are slightly
        larger than what is requested
        '''
        cdef size_t min_requested = size
        cdef size_t max_requested = size * 2 + 4096
        cdef size_t idx, i
        cdef GPUHeap p

        with self._rlock:
            device = cuGetDevice()
            pools = self._pool_lists[device]

            idx = bisect.bisect_left(pools, (size,))


            for i in range(idx, len(pools)):
                _, p = pools[i]
                if p.nbytes >= max_requested:
                    break

                if min_requested <= p.nbytes and heapReady(p):
                    pool = p
                    del pools[i]
                    break

        return pool

    cpdef release_pool(self, deviceID=None):
        if deviceID is None:
            self._pool_lists = collections.defaultdict(list)
        else:
            del self._pool_lists[deviceID]


gpu_allocator = GpuAllocator()

cdef GpuAllocator c_gpu_allocator
c_gpu_allocator = gpu_allocator

cpdef GpuAllocator get_gpu_allocator():
    return c_gpu_allocator


cpdef _cuSetLimit(limit, value):
    cdef size_t c_value=999;

    cuInit(0)

    ret = cuCtxGetLimit(&c_value, limit)

    cuCtxSetLimit(limit, value)
