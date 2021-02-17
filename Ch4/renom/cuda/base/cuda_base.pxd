from libc.stdint cimport uintptr_t

ctypedef uintptr_t cudaStream_ptr

cdef extern from "cuda_runtime.h":
    ctypedef enum cudaError_t:
      cudaSuccess                           =      0,
      cudaErrorMissingConfiguration         =      1,
      cudaErrorMemoryAllocation             =      2,
      cudaErrorInitializationError          =      3,
      cudaErrorLaunchFailure                =      4,
      cudaErrorPriorLaunchFailure           =      5,
      cudaErrorLaunchTimeout                =      6,
      cudaErrorLaunchOutOfResources         =      7,
      cudaErrorInvalidDeviceFunction        =      8,
      cudaErrorInvalidConfiguration         =      9,
      cudaErrorInvalidDevice                =     10,
      cudaErrorInvalidValue                 =     11,
      cudaErrorInvalidPitchValue            =     12,
      cudaErrorInvalidSymbol                =     13,
      cudaErrorMapBufferObjectFailed        =     14,
      cudaErrorUnmapBufferObjectFailed      =     15,
      cudaErrorInvalidHostPointer           =     16,
      cudaErrorInvalidDevicePointer         =     17,
      cudaErrorInvalidTexture               =     18,
      cudaErrorInvalidTextureBinding        =     19,
      cudaErrorInvalidChannelDescriptor     =     20,
      cudaErrorInvalidMemcpyDirection       =     21,
      cudaErrorAddressOfConstant            =     22,
      cudaErrorTextureFetchFailed           =     23,
      cudaErrorTextureNotBound              =     24,
      cudaErrorSynchronizationError         =     25,
      cudaErrorInvalidFilterSetting         =     26,
      cudaErrorInvalidNormSetting           =     27,
      cudaErrorMixedDeviceExecution         =     28,
      cudaErrorCudartUnloading              =     29,
      cudaErrorUnknown                      =     30,
      cudaErrorNotYetImplemented            =     31,
      cudaErrorMemoryValueTooLarge          =     32,
      cudaErrorInvalidResourceHandle        =     33,
      cudaErrorNotReady                     =     34,
      cudaErrorInsufficientDriver           =     35,
      cudaErrorSetOnActiveProcess           =     36,
      cudaErrorInvalidSurface               =     37,
      cudaErrorNoDevice                     =     38,
      cudaErrorECCUncorrectable             =     39,
      cudaErrorSharedObjectSymbolNotFound   =     40,
      cudaErrorSharedObjectInitFailed       =     41,
      cudaErrorUnsupportedLimit             =     42,
      cudaErrorDuplicateVariableName        =     43,
      cudaErrorDuplicateTextureName         =     44,
      cudaErrorDuplicateSurfaceName         =     45,
      cudaErrorDevicesUnavailable           =     46,
      cudaErrorInvalidKernelImage           =     47,
      cudaErrorNoKernelImageForDevice       =     48,
      cudaErrorIncompatibleDriverContext    =     49,
      cudaErrorPeerAccessAlreadyEnabled     =     50,
      cudaErrorPeerAccessNotEnabled         =     51,
      cudaErrorDeviceAlreadyInUse           =     54,
      cudaErrorProfilerDisabled             =     55,
      cudaErrorProfilerNotInitialized       =     56,
      cudaErrorProfilerAlreadyStarted       =     57,
      cudaErrorProfilerAlreadyStopped       =     58,
      cudaErrorAssert                       =     59,
      cudaErrorTooManyPeers                 =     60,
      cudaErrorHostMemoryAlreadyRegistered  =     61,
      cudaErrorHostMemoryNotRegistered      =     62,
      cudaErrorOperatingSystem              =     63,
      cudaErrorPeerAccessUnsupported        =     64,
      cudaErrorLaunchMaxDepthExceeded       =     65,
      cudaErrorLaunchFileScopedTex          =     66,
      cudaErrorLaunchFileScopedSurf         =     67,
      cudaErrorSyncDepthExceeded            =     68,
      cudaErrorLaunchPendingCountExceeded   =     69,
      cudaErrorNotPermitted                 =     70,
      cudaErrorNotSupported                 =     71,
      cudaErrorHardwareStackError           =     72,
      cudaErrorIllegalInstruction           =     73,
      cudaErrorMisalignedAddress            =     74,
      cudaErrorInvalidAddressSpace          =     75,
      cudaErrorInvalidPc                    =     76,
      cudaErrorIllegalAddress               =     77,
      cudaErrorInvalidPtx                   =     78,
      cudaErrorInvalidGraphicsContext       =     79,
      cudaErrorNvlinkUncorrectable          =     80,
      cudaErrorJitCompilerNotFound          =     81,
      cudaErrorCooperativeLaunchTooLarge    =     82,
      cudaErrorStartupFailure               =   0x7f,
      cudaErrorApiFailureBase               =  10000

    ctypedef struct cudaDeviceProp:
      char name[256],
      size_t totalGlobalMem,
      size_t sharedMemPerBlock,
      int regsPerBlock,
      int warpSize,
      size_t memPitch,
      int maxThreadsPerBlock,
      int maxThreadsDim[3],
      int maxGridSize[3],
      int clockRate,
      size_t totalConstMem,
      int major,
      int minor,
      size_t textureAlignment,
      size_t texturePitchAlignment,
      int deviceOverlap,
      int multiProcessorCount,
      int kernelExecTimeoutEnabled,
      int integrated,
      int canMapHostMemory,
      int computeMode,
      int maxTexture1D,
      int maxTexture1DMipmap,
      int maxTexture1DLinear,
      int maxTexture2D[2],
      int maxTexture2DMipmap[2],
      int maxTexture2DLinear[3],
      int maxTexture2DGather[2],
      int maxTexture3D[3],
      int maxTexture3DAlt[3],
      int maxTextureCubemap,
      int maxTexture1DLayered[2],
      int maxTexture2DLayered[3],
      int maxTextureCubemapLayered[2],
      int maxSurface1D,
      int maxSurface2D[2],
      int maxSurface3D[3],
      int maxSurface1DLayered[2],
      int maxSurface2DLayered[3],
      int maxSurfaceCubemap,
      int maxSurfaceCubemapLayered[2],
      size_t surfaceAlignment,
      int concurrentKernels,
      int ECCEnabled,
      int pciBusID,
      int pciDeviceID,
      int pciDomainID,
      int tccDriver,
      int asyncEngineCount,
      int unifiedAddressing,
      int memoryClockRate,
      int memoryBusWidth,
      int l2CacheSize,
      int maxThreadsPerMultiProcessor,
      int streamPrioritiesSupported,
      int globalL1CacheSupported,
      int localL1CacheSupported,
      size_t sharedMemPerMultiprocessor,
      int regsPerMultiprocessor,
      int managedMemory,
      int isMultiGpuBoard,
      int multiGpuBoardGroupID,
      int singleToDoublePrecisionPerfRatio,
      int pageableMemoryAccess,
      int concurrentManagedAccess,
      int computePreemptionSupported,
      int canUseHostPointerForRegisteredMem,
      int cooperativeLaunch,
      int cooperativeMultiDeviceLaunch,
      int pageableMemoryAccessUsesHostPageTables,
      int directManagedMemAccessFromHost,



    ctypedef struct CUevent_st:
      pass
    ctypedef struct CUstream_st:
      pass

    ctypedef CUevent_st * cudaEvent_t
    ctypedef CUstream_st *  cudaStream_t

    ctypedef enum cudaMemcpyKind:
        cudaMemcpyHostToHost,
        cudaMemcpyHostToDevice,
        cudaMemcpyDeviceToHost,
        cudaMemcpyDeviceToDevice
    ctypedef enum cudaStreamFlags:
        cudaStreamDefault = 0x00
        cudaStreamLegacy = 0x1
        cudaStreamNonBlocking = 0x01
        cudaStreamPerThread = 0x2
    ctypedef int size_t

    cudaError_t cudaMemset(void * devPtr, int value, size_t size)
    cudaError_t cudaMemcpy(void * dst, const void * src, int size, cudaMemcpyKind kind)
    cudaError_t cudaMemcpyAsync(void * dst, const void * src, int count, cudaMemcpyKind kind, cudaStream_t stream)
    cudaError_t cudaMemcpyPeer(void * dst, int dstDevice, const void * src, int srcDevice, size_t count)
    cudaError_t cudaMemGetInfo(size_t * free, size_t * total)

    cudaError_t cudaHostAlloc(void ** ptr, size_t size, int flags)
    cudaError_t cudaMallocHost(void ** ptr, size_t size)
    cudaError_t cudaFreeHost(void * ptr)
    cudaError_t cudaHostRegister(void * ptr, size_t size, int flags)
    cudaError_t cudaHostUnregister(void *ptr)
    cudaError_t cudaEventCreate(cudaEvent_t * event)
    cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
    cudaError_t cudaEventSynchronize(cudaEvent_t)
    cudaError_t cudaEventQuery(cudaEvent_t)
    cudaError_t cudaMalloc(void ** ptr, size_t size)
    #cudaError_t cudaMallocHost(void ** ptr, size_t size, unsigned int flags)
    cudaError_t cudaSetDevice(int size)
    cudaError_t cudaGetDevice(int *device)
    cudaError_t cudaDeviceReset()
    cudaError_t cudaDeviceSynchronize()
    cudaError_t cudaFree(void * ptr)
    const char * cudaGetErrorString(cudaError_t erorr)

    cudaError_t cudaStreamCreate(cudaStream_t * pStream)
    cudaError_t cudaStreamCreateWithFlags(cudaStream_t * pStream, unsigned int flags)
    cudaError_t cudaStreamDestroy(cudaStream_t stream)
    cudaError_t cudaStreamSynchronize(cudaStream_t stream)
    cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned flags)

    cudaError_t cudaGetDeviceProperties(cudaDeviceProp * prop, int device)
    cudaError_t cudaGetDeviceCount(int * count)
    cudaError_t cudaDeviceCanAccessPeer( int* canAccessPeer, int  device, int  peerDevice)


cdef extern from "cuda.h":
    ctypedef enum cudaError_enum:
        CUDA_SUCCESS,

    ctypedef enum CUlimit:
        CU_LIMIT_STACK_SIZE, CU_LIMIT_PRINTF_FIFO_SIZE, CU_LIMIT_MALLOC_HEAP_SIZE,
        CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH, CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT,
        CU_LIMIT_MAX

    ctypedef cudaError_enum CUresult
    ctypedef int CUcontext
    ctypedef int CUdevice

    CUresult cuCtxCreate(CUcontext * pctx, unsigned int flags, CUdevice dev)
    CUresult cuCtxDestroy(CUcontext ctx)
    CUresult cuCtxGetDevice(CUdevice * device)
    CUresult cuGetErrorString(CUresult error, const char ** pStr)

    CUresult cuCtxSetLimit (CUlimit limit, size_t value)
    CUresult cuCtxGetLimit (size_t *value, CUlimit limit)

    CUresult cuInit(unsigned int)

cdef extern from "nvToolsExtCudaRt.h":
  void nvtxNameCudaStreamA(cudaStream_t stream, const char* name)

cdef class GPUHeap(object):
    cpdef object _mystream
    cdef public size_t ptr
    cdef public size_t nbytes
    cdef public int device_id
    cdef public int refcount
    cdef cudaEvent_t event

    cpdef memcpyH2D(self, cpu_ptr, size_t nbytes)
    cpdef memcpyD2H(self, cpu_ptr, size_t nbytes)
    cpdef memcpyD2D(self, gpu_ptr, size_t nbytes)
    cpdef copy_from(self, other, size_t nbytes)


cdef class GpuAllocator(object):
    cpdef object _pool_lists, _all_pools
    cpdef object _memsync_stream
    cpdef object _rlock

    cpdef GPUHeap malloc(self, size_t nbytes)
    cpdef GPUHeap getAvailablePool(self, size_t size)
    cpdef free(self, GPUHeap pool)
    cpdef release_pool(self, deviceID=*)


cpdef GpuAllocator get_gpu_allocator()

cdef GpuAllocator c_gpu_allocator
