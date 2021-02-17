from renom.cuda.base.cuda_base cimport *

cdef extern from "cudnn.h":
    ctypedef int size_t
    ctypedef struct cudnnContext
    ctypedef cudnnContext * cudnnHandle_t
    ctypedef enum cudnnStatus_t:

        CUDNN_STATUS_SUCCESS,
        CUDNN_STATUS_NOT_INITIALIZED,
        CUDNN_STATUS_ALLOC_FAILED,
        CUDNN_STATUS_BAD_PARAM,
        CUDNN_STATUS_INTERNAL_ERROR,
        CUDNN_STATUS_INVALID_VALUE,
        CUDNN_STATUS_ARCH_MISMATCH,
        CUDNN_STATUS_MAPPING_ERROR,
        CUDNN_STATUS_EXECUTION_FAILED,
        CUDNN_STATUS_NOT_SUPPORTED,
        CUDNN_STATUS_LICENSE_ERROR

    size_t cudnnGetVersion()

    # human-readable error messages
    const char * cudnnGetErrorString(cudnnStatus_t status)

    cudnnStatus_t cudnnCreate(cudnnHandle_t * handle)
    cudnnStatus_t cudnnDestroy(cudnnHandle_t handle)
    cudnnStatus_t cudnnSetStream(cudnnHandle_t handle, cudaStream_t streamId)
    cudnnStatus_t cudnnGetStream(cudnnHandle_t handle, cudaStream_t * streamId)

    # Data structures to represent Image/Filter and the Neural Network Layer

    ctypedef struct cudnnTensorStruct
    ctypedef struct cudnnConvolutionStruct
    ctypedef struct cudnnPoolingStruct
    ctypedef struct cudnnFilterStruct
    ctypedef struct cudnnLRNStruct
    ctypedef struct cudnnActivationStruct
    ctypedef struct cudnnSpatialTransformerStruct
    ctypedef struct cudnnOpTensorStruct

    ctypedef cudnnTensorStruct * cudnnTensorDescriptor_t
    ctypedef cudnnConvolutionStruct * cudnnConvolutionDescriptor_t
    ctypedef cudnnPoolingStruct * cudnnPoolingDescriptor_t
    ctypedef cudnnFilterStruct * cudnnFilterDescriptor_t
    ctypedef cudnnLRNStruct * cudnnLRNDescriptor_t
    ctypedef cudnnActivationStruct * cudnnActivationDescriptor_t
    ctypedef cudnnSpatialTransformerStruct * cudnnSpatialTransformerDescriptor_t
    ctypedef cudnnOpTensorStruct * cudnnOpTensorDescriptor_t

    # CUDNN data type
    ctypedef enum cudnnDataType_t:
        CUDNN_DATA_FLOAT,
        CUDNN_DATA_DOUBLE,
        CUDNN_DATA_HALF

    # CUDNN propagate Nan
    ctypedef enum cudnnNanPropagation_t:
        CUDNN_NOT_PROPAGATE_NAN,
        CUDNN_PROPAGATE_NAN

    # Maximum supported number of tensor dimensions
    # define CUDNN_DIM_MAX 8

    # Create an instance of a generic Tensor descriptor
    cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t * tensorDesc)

    ctypedef enum cudnnTensorFormat_t:
        CUDNN_TENSOR_NCHW,      # row major (wStride = 1, hStride = w)
        CUDNN_TENSOR_NHWC       # feature maps interleaved ( cStride = 1 )

    cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc,
                                             cudnnTensorFormat_t format,
                                             cudnnDataType_t dataType,   # image data type
                                             # number of inputs (batch size)
                                             int n,
                                             int c,                      # number of input feature maps
                                             int h,                      # height of input section
                                             int w)                     # width of input section

    cudnnStatus_t cudnnSetTensor4dDescriptorEx(
        cudnnTensorDescriptor_t tensorDesc,
        cudnnDataType_t dataType,   # image data type
        int n,                      # number of inputs (batch size)
        int c,                      # number of input feature maps
        int h,                      # height of input section
        int w,                      # width of input section
        int nStride,
        int cStride,
        int hStride,
        int wStride)

    cudnnStatus_t cudnnGetTensor4dDescriptor(
        const cudnnTensorDescriptor_t tensorDesc,
        cudnnDataType_t * dataType,  # image data type
        int * n,        # number of inputs (batch size)
        int * c,        # number of input feature maps
        int * h,        # height of input section
        int * w,        # width of input section
        int * nStride,
        int * cStride,
        int * hStride,
        int * wStride)

    cudnnStatus_t cudnnSetTensorNdDescriptor(
        cudnnTensorDescriptor_t tensorDesc,
        cudnnDataType_t dataType,
        int nbDims,
        const int dimA[],
        const int strideA[])

    cudnnStatus_t cudnnSetTensorNdDescriptorEx(
        cudnnTensorDescriptor_t tensorDesc,
        cudnnTensorFormat_t format,
        cudnnDataType_t dataType,
        int nbDims,
        const int dimA[])

    cudnnStatus_t cudnnGetTensorNdDescriptor(
        const cudnnTensorDescriptor_t tensorDesc,
        int nbDimsRequested,
        cudnnDataType_t * dataType,
        int * nbDims,
        int dimA[],
        int strideA[])

    cudnnStatus_t cudnnDestroyTensorDescriptor(
        cudnnTensorDescriptor_t  tensorDesc)

    cudnnStatus_t cudnnTransformTensor(
        cudnnHandle_t handle,
        const void * alpha,
        const cudnnTensorDescriptor_t xDesc,
        const void * x,
        const void * beta,
        const cudnnTensorDescriptor_t yDesc,
        void * y)

    cudnnStatus_t cudnnAddTensor(
        cudnnHandle_t handle,
        const void * alpha,
        const cudnnTensorDescriptor_t aDesc,
        const void * A,
        const void * beta,
        const cudnnTensorDescriptor_t cDesc,
        void * C)

    ctypedef enum cudnnOpTensorOp_t:
        CUDNN_OP_TENSOR_ADD,
        CUDNN_OP_TENSOR_MUL,
        CUDNN_OP_TENSOR_MIN,
        CUDNN_OP_TENSOR_MAX

    cudnnStatus_t cudnnCreateOpTensorDescriptor(
        cudnnOpTensorDescriptor_t * opTensorDesc)

    cudnnStatus_t cudnnSetOpTensorDescriptor(
        cudnnOpTensorDescriptor_t opTensorDesc,
        cudnnOpTensorOp_t opTensorOp,
        cudnnDataType_t opTensorCompType,
        cudnnNanPropagation_t opTensorNanOpt)

    cudnnStatus_t cudnnGetOpTensorDescriptor(
        const cudnnOpTensorDescriptor_t opTensorDesc,
        cudnnOpTensorOp_t * opTensorOp,
        cudnnDataType_t * opTensorCompType,
        cudnnNanPropagation_t * opTensorNanOpt)

    cudnnStatus_t cudnnDestroyOpTensorDescriptor(
        cudnnOpTensorDescriptor_t opTensorDesc)

    # Tensor Bias operation : C = op( alpha1 * A, alpha2 * B ) + beta * C
    cudnnStatus_t cudnnOpTensor(
        cudnnHandle_t handle,
        const cudnnOpTensorDescriptor_t opTensorDesc,
        const void * alpha1,
        const cudnnTensorDescriptor_t aDesc,
        const void * A,
        const void * alpha2,
        const cudnnTensorDescriptor_t bDesc,
        const void * B,
        const void * beta,
        const cudnnTensorDescriptor_t cDesc,
        void * C)

    # Set all values of a tensor to a given value : y[i] = value[0]
    cudnnStatus_t cudnnSetTensor(
        cudnnHandle_t handle,
        const cudnnTensorDescriptor_t yDesc,
        void * y,
        const void * valuePtr)

    # Scale all values of a tensor by a given factor : y[i] = alpha * y[i]
    cudnnStatus_t cudnnScaleTensor(
        cudnnHandle_t handle,
        const cudnnTensorDescriptor_t yDesc,
        void * y,
        const void * alpha)

    # convolution mode
    ctypedef enum cudnnConvolutionMode_t:
        CUDNN_CONVOLUTION,
        CUDNN_CROSS_CORRELATION

    # Create an instance of FilterStruct
    cudnnStatus_t cudnnCreateFilterDescriptor(
        cudnnFilterDescriptor_t * filterDesc)

    cudnnStatus_t cudnnSetFilter4dDescriptor(
        cudnnFilterDescriptor_t filterDesc,
        cudnnDataType_t dataType,  # image data type
        cudnnTensorFormat_t format,
        int k,        # number of output feature maps
        int c,        # number of input feature maps
        int h,        # height of each input filter
        int w)      # width of  each input filter

    cudnnStatus_t cudnnGetFilter4dDescriptor(
        const cudnnFilterDescriptor_t filterDesc,
        cudnnDataType_t * dataType,  # image data type
        cudnnTensorFormat_t * format,
        int * k,        # number of output feature maps
        int * c,        # number of input feature maps
        int * h,        # height of each input filter
        int * w)      # width of  each input filter

    cudnnStatus_t cudnnSetFilterNdDescriptor(
        cudnnFilterDescriptor_t filterDesc,
        cudnnDataType_t dataType,  # image data type
        cudnnTensorFormat_t format,
        int nbDims,
        const int filterDimA[])

    cudnnStatus_t cudnnGetFilterNdDescriptor(
        const cudnnFilterDescriptor_t filterDesc,
        int nbDimsRequested,
        cudnnDataType_t * dataType,  # image data type
        cudnnTensorFormat_t * format,
        int * nbDims,
        int filterDimA[])

    cudnnStatus_t cudnnDestroyFilterDescriptor(
        cudnnFilterDescriptor_t filterDesc)

    cudnnStatus_t cudnnCreateConvolutionDescriptor(
        cudnnConvolutionDescriptor_t * convDesc)

    cudnnStatus_t cudnnGetConvolution2dDescriptor(
        const cudnnConvolutionDescriptor_t convDesc,
        int * pad_h,    # zero-padding height
        int * pad_w,    # zero-padding width
        int * u,        # vertical filter stride
        int * v,        # horizontal filter stride
        int * upscalex,  # upscale the input in x-direction
        int * upscaley,  # upscale the input in y-direction
        cudnnConvolutionMode_t * mode)

    cudnnStatus_t cudnnGetConvolution2dDescriptor_v5(
        const cudnnConvolutionDescriptor_t convDesc,
        int * pad_h,    # zero-padding height
        int * pad_w,    # zero-padding width
        int * u,        # vertical filter stride
        int * v,        # horizontal filter stride
        int * upscalex,  # upscale the input in x-direction
        int * upscaley,  # upscale the input in y-direction
        cudnnConvolutionMode_t * mode,
        cudnnDataType_t * dataType)

    cudnnStatus_t cudnnGetConvolution2dForwardOutputDim(
        const cudnnConvolutionDescriptor_t  convDesc,
        const cudnnTensorDescriptor_t inputTensorDesc,
        const cudnnFilterDescriptor_t filterDesc,
        int * n,
        int * c,
        int * h,
        int * w)

    cudnnStatus_t cudnnSetConvolution2dDescriptor(
        cudnnConvolutionDescriptor_t    convDesc,
        int                             pad_h,
        int                             pad_w,
        int                             u,
        int                             v,
        int                             dilation_h,
        int                             dilation_w,
        cudnnConvolutionMode_t          mode,
        cudnnDataType_t                 computeType)

    cudnnStatus_t cudnnSetConvolutionGroupCount(
        cudnnConvolutionDescriptor_t    convDesc,
        int                             groupCount)

    cudnnStatus_t cudnnSetConvolutionNdDescriptor(
        cudnnConvolutionDescriptor_t convDesc,
        int arrayLength,             # nbDims-2 size
        const int padA[],
        const int filterStrideA[],
        const int upscaleA[],
        cudnnConvolutionMode_t mode,
        cudnnDataType_t dataType)  # convolution data type

    cudnnStatus_t cudnnGetConvolutionNdDescriptor(
        const cudnnConvolutionDescriptor_t convDesc,
        int arrayLengthRequested,
        int * arrayLength,
        int padA[],
        int strideA[],
        int upscaleA[],
        cudnnConvolutionMode_t * mode,
        cudnnDataType_t * dataType)   # convolution data type

    cudnnStatus_t cudnnGetConvolutionNdForwardOutputDim(
        const cudnnConvolutionDescriptor_t convDesc,
        const cudnnTensorDescriptor_t inputTensorDesc,
        const cudnnFilterDescriptor_t filterDesc,
        int nbDims,
        int tensorOuputDimA[])

    cudnnStatus_t cudnnDestroyConvolutionDescriptor(
        cudnnConvolutionDescriptor_t convDesc)

    ctypedef enum cudnnConvolutionFwdPreference_t:
        CUDNN_CONVOLUTION_FWD_NO_WORKSPACE,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT

    ctypedef enum cudnnConvolutionFwdAlgo_t:
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
        CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
        CUDNN_CONVOLUTION_FWD_ALGO_FFT,
        CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED

    ctypedef struct cudnnConvolutionFwdAlgoPerf_t:
        cudnnConvolutionFwdAlgo_t algo
        cudnnStatus_t status
        float time
        size_t memory

    cudnnStatus_t cudnnFindConvolutionForwardAlgorithm(
        cudnnHandle_t handle,
        const cudnnTensorDescriptor_t xDesc,
        const cudnnFilterDescriptor_t wDesc,
        const cudnnConvolutionDescriptor_t convDesc,
        const cudnnTensorDescriptor_t yDesc,
        const int requestedAlgoCount,
        int * returnedAlgoCount,
        cudnnConvolutionFwdAlgoPerf_t * perfResults)

    cudnnStatus_t cudnnFindConvolutionForwardAlgorithmEx(
        cudnnHandle_t handle,
        const cudnnTensorDescriptor_t xDesc,
        const void * x,
        const cudnnFilterDescriptor_t wDesc,
        const void * w,
        const cudnnConvolutionDescriptor_t convDesc,
        const cudnnTensorDescriptor_t yDesc,
        void * y,
        const int requestedAlgoCount,
        int * returnedAlgoCount,
        cudnnConvolutionFwdAlgoPerf_t * perfResults,
        void * workSpace,
        size_t workSpaceSizeInBytes)

    cudnnStatus_t cudnnGetConvolutionForwardAlgorithm(
        cudnnHandle_t handle,
        const cudnnTensorDescriptor_t xDesc,
        const cudnnFilterDescriptor_t wDesc,
        const cudnnConvolutionDescriptor_t convDesc,
        const cudnnTensorDescriptor_t yDesc,
        cudnnConvolutionFwdPreference_t preference,
        size_t memoryLimitInBytes,
        cudnnConvolutionFwdAlgo_t * algo)

    # convolution algorithm (which requires potentially some workspace)

    # Helper function to return the minimum size of the workspace to be passed
    # to the convolution given an algo
    cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(
        cudnnHandle_t handle,
        const cudnnTensorDescriptor_t xDesc,
        const cudnnFilterDescriptor_t wDesc,
        const cudnnConvolutionDescriptor_t convDesc,
        const cudnnTensorDescriptor_t yDesc,
        cudnnConvolutionFwdAlgo_t algo,
        size_t * sizeInBytes)

    cudnnStatus_t cudnnConvolutionForward(
        cudnnHandle_t handle,
        const void * alpha,
        const cudnnTensorDescriptor_t xDesc,
        const void * x,
        const cudnnFilterDescriptor_t wDesc,
        const void * w,
        const cudnnConvolutionDescriptor_t convDesc,
        cudnnConvolutionFwdAlgo_t algo,
        void * workSpace,
        size_t workSpaceSizeInBytes,
        const void * beta,
        const cudnnTensorDescriptor_t yDesc,
        void * y)

    cudnnStatus_t cudnnConvolutionBackwardBias(
        cudnnHandle_t handle,
        const void * alpha,
        const cudnnTensorDescriptor_t dyDesc,
        const void * dy,
        const void * beta,
        const cudnnTensorDescriptor_t dbDesc,
        void * db)

    ctypedef enum cudnnConvolutionBwdFilterPreference_t:
        CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE,
        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
        CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT

    ctypedef enum cudnnConvolutionBwdFilterAlgo_t:
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,  # non-deterministic
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3,  # non-deterministic, algo0 with workspace
        # CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD, # not implemented
        CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED

    ctypedef struct cudnnConvolutionBwdFilterAlgoPerf_t:
        cudnnConvolutionBwdFilterAlgo_t algo
        cudnnStatus_t status
        float time
        size_t memory

    cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithm(
        cudnnHandle_t handle,
        const cudnnTensorDescriptor_t xDesc,
        const cudnnTensorDescriptor_t dyDesc,
        const cudnnConvolutionDescriptor_t convDesc,
        const cudnnFilterDescriptor_t dwDesc,
        const int requestedAlgoCount,
        int * returnedAlgoCount,
        cudnnConvolutionBwdFilterAlgoPerf_t * perfResults)

    cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithmEx(
        cudnnHandle_t handle,
        const cudnnTensorDescriptor_t xDesc,
        const void * x,
        const cudnnTensorDescriptor_t dyDesc,
        const void * y,
        const cudnnConvolutionDescriptor_t convDesc,
        const cudnnFilterDescriptor_t dwDesc,
        void * dw,
        const int requestedAlgoCount,
        int * returnedAlgoCount,
        cudnnConvolutionBwdFilterAlgoPerf_t * perfResults,
        void * workSpace,
        size_t workSpaceSizeInBytes)

    cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm(
        cudnnHandle_t handle,
        const cudnnTensorDescriptor_t xDesc,
        const cudnnTensorDescriptor_t dyDesc,
        const cudnnConvolutionDescriptor_t convDesc,
        const cudnnFilterDescriptor_t dwDesc,
        cudnnConvolutionBwdFilterPreference_t preference,
        size_t memoryLimitInBytes,
        cudnnConvolutionBwdFilterAlgo_t * algo)

    # convolution algorithm (which requires potentially some workspace)

    # Helper function to return the minimum size of the workspace to be passed
    # to the convolution given an algo
    cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(
        cudnnHandle_t handle,
        const cudnnTensorDescriptor_t xDesc,
        const cudnnTensorDescriptor_t dyDesc,
        const cudnnConvolutionDescriptor_t convDesc,
        const cudnnFilterDescriptor_t gradDesc,
        cudnnConvolutionBwdFilterAlgo_t algo,
        size_t * sizeInBytes)

    cudnnStatus_t cudnnConvolutionBackwardFilter(
        cudnnHandle_t handle,
        const void * alpha,
        const cudnnTensorDescriptor_t xDesc,
        const void * x,
        const cudnnTensorDescriptor_t dyDesc,
        const void * dy,
        const cudnnConvolutionDescriptor_t convDesc,
        cudnnConvolutionBwdFilterAlgo_t algo,
        void * workSpace,
        size_t workSpaceSizeInBytes,
        const void * beta,
        const cudnnFilterDescriptor_t dwDesc,
        void * dw)

    ctypedef enum cudnnConvolutionBwdDataPreference_t:
        CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE,
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
        CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT

    ctypedef enum cudnnConvolutionBwdDataAlgo_t:
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,  # non-deterministic
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED

    ctypedef struct cudnnConvolutionBwdDataAlgoPerf_t:
        cudnnConvolutionBwdDataAlgo_t algo
        cudnnStatus_t status
        float time
        size_t memory

    cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithm(
        cudnnHandle_t handle,
        const cudnnFilterDescriptor_t wDesc,
        const cudnnTensorDescriptor_t dyDesc,
        const cudnnConvolutionDescriptor_t convDesc,
        const cudnnTensorDescriptor_t dxDesc,
        const int requestedAlgoCount,
        int * returnedAlgoCount,
        cudnnConvolutionBwdDataAlgoPerf_t * perfResults)

    cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithmEx(
        cudnnHandle_t handle,
        const cudnnFilterDescriptor_t wDesc,
        const void * w,
        const cudnnTensorDescriptor_t dyDesc,
        const void * dy,
        const cudnnConvolutionDescriptor_t convDesc,
        const cudnnTensorDescriptor_t dxDesc,
        void * dx,
        const int requestedAlgoCount,
        int * returnedAlgoCount,
        cudnnConvolutionBwdDataAlgoPerf_t * perfResults,
        void * workSpace,
        size_t workSpaceSizeInBytes)

    cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm(
        cudnnHandle_t handle,
        const cudnnFilterDescriptor_t wDesc,
        const cudnnTensorDescriptor_t dyDesc,
        const cudnnConvolutionDescriptor_t convDesc,
        const cudnnTensorDescriptor_t dxDesc,
        cudnnConvolutionBwdDataPreference_t preference,
        size_t memoryLimitInBytes,
        cudnnConvolutionBwdDataAlgo_t * algo)

    cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnnHandle_t handle,
        const cudnnFilterDescriptor_t wDesc,
        const cudnnTensorDescriptor_t dyDesc,
        const cudnnConvolutionDescriptor_t convDesc,
        const cudnnTensorDescriptor_t dxDesc,
        cudnnConvolutionBwdDataAlgo_t algo,
        size_t * sizeInBytes)

    cudnnStatus_t cudnnConvolutionBackwardData(
        cudnnHandle_t handle,
        const void * alpha,
        const cudnnFilterDescriptor_t wDesc,
        const void * w,
        const cudnnTensorDescriptor_t dyDesc,
        const void * dy,
        const cudnnConvolutionDescriptor_t convDesc,
        cudnnConvolutionBwdDataAlgo_t algo,
        void * workSpace,
        size_t workSpaceSizeInBytes,
        const void * beta,
        const cudnnTensorDescriptor_t dxDesc,
        void * dx)

    cudnnStatus_t cudnnIm2Col(
        cudnnHandle_t handle,
        const cudnnTensorDescriptor_t xDesc,
        const void * x,
        const cudnnFilterDescriptor_t wDesc,
        const cudnnConvolutionDescriptor_t convDesc,
        void * colBuffer)

    # softmax algorithm
    ctypedef enum cudnnSoftmaxAlgorithm_t:

        CUDNN_SOFTMAX_FAST,         # straightforward implementation
        CUDNN_SOFTMAX_ACCURATE,     # subtract max from every point to avoid overflow
        CUDNN_SOFTMAX_LOG

    ctypedef enum cudnnSoftmaxMode_t:
        CUDNN_SOFTMAX_MODE_INSTANCE,    # compute the softmax over all C, H, W for each N
        CUDNN_SOFTMAX_MODE_CHANNEL      # compute the softmax over all C for each H, W, N

    cudnnStatus_t cudnnSoftmaxForward(
        cudnnHandle_t handle,
        cudnnSoftmaxAlgorithm_t algo,
        cudnnSoftmaxMode_t mode,
        const void * alpha,
        const cudnnTensorDescriptor_t xDesc,
        const void * x,
        const void * beta,
        const cudnnTensorDescriptor_t yDesc,
        void * y)

    cudnnStatus_t cudnnSoftmaxBackward(
        cudnnHandle_t handle,
        cudnnSoftmaxAlgorithm_t algo,
        cudnnSoftmaxMode_t mode,
        const void * alpha,
        const cudnnTensorDescriptor_t yDesc,
        const void * y,
        const cudnnTensorDescriptor_t dyDesc,
        const void * dy,
        const void * beta,
        const cudnnTensorDescriptor_t dxDesc,
        void * dx)

    # pooling mode
    ctypedef enum cudnnPoolingMode_t:
        CUDNN_POOLING_MAX,
        # count for average includes padded values
        CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
        # count for average does not include padded values
        CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING

    cudnnStatus_t cudnnCreatePoolingDescriptor(
        cudnnPoolingDescriptor_t * poolingDesc)

    cudnnStatus_t cudnnSetPooling2dDescriptor(
        cudnnPoolingDescriptor_t            poolingDesc,
        cudnnPoolingMode_t                  mode,
        cudnnNanPropagation_t               maxpoolingNanOpt,
        int                                 windowHeight,
        int                                 windowWidth,
        int                                 verticalPadding,
        int                                 horizontalPadding,
        int                                 verticalStride,
        int                                 horizontalStride)

    cudnnStatus_t cudnnGetPooling2dDescriptor(
        const cudnnPoolingDescriptor_t      poolingDesc,
        cudnnPoolingMode_t * mode,
        cudnnNanPropagation_t * maxpoolingNanOpt,
        int * windowHeight,
        int * windowWidth,
        int * verticalPadding,
        int * horizontalPadding,
        int * verticalStride,
        int * horizontalStride)

    cudnnStatus_t cudnnSetPoolingNdDescriptor(
        cudnnPoolingDescriptor_t            poolingDesc,
        const cudnnPoolingMode_t            mode,
        const cudnnNanPropagation_t         maxpoolingNanOpt,
        int                                 nbDims,
        const int                           windowDimA[],
        const int                           paddingA[],
        const int                           strideA[])

    cudnnStatus_t cudnnGetPoolingNdDescriptor(
        const cudnnPoolingDescriptor_t      poolingDesc,
        int                                 nbDimsRequested,
        cudnnPoolingMode_t * mode,
        cudnnNanPropagation_t * maxpoolingNanOpt,
        int * nbDims,
        int                                 windowDimA[],
        int                                 paddingA[],
        int                                 strideA[])

    cudnnStatus_t cudnnGetPoolingNdForwardOutputDim(
        const cudnnPoolingDescriptor_t      poolingDesc,
        const cudnnTensorDescriptor_t       inputTensorDesc,
        int                                 nbDims,
        int                                 outputTensorDimA[])

    cudnnStatus_t cudnnGetPooling2dForwardOutputDim(
        const cudnnPoolingDescriptor_t      poolingDesc,
        const cudnnTensorDescriptor_t       inputTensorDesc,
        int * n,
        int * c,
        int * h,
        int * w)

    cudnnStatus_t cudnnDestroyPoolingDescriptor(
        cudnnPoolingDescriptor_t            poolingDesc)

    cudnnStatus_t cudnnPoolingForward(
        cudnnHandle_t                       handle,
        const cudnnPoolingDescriptor_t      poolingDesc,
        const void * alpha,
        const cudnnTensorDescriptor_t       xDesc,
        const void * x,
        const void * beta,
        const cudnnTensorDescriptor_t       yDesc,
        void * y)

    cudnnStatus_t cudnnPoolingBackward(
        cudnnHandle_t                       handle,
        const cudnnPoolingDescriptor_t      poolingDesc,
        const void * alpha,
        const cudnnTensorDescriptor_t       yDesc,
        const void * y,
        const cudnnTensorDescriptor_t       dyDesc,
        const void * dy,
        const cudnnTensorDescriptor_t       xDesc,
        const void * x,
        const void * beta,
        const cudnnTensorDescriptor_t       dxDesc,
        void * dx)

    ctypedef enum cudnnActivationMode_t:
        CUDNN_ACTIVATION_SIGMOID,
        CUDNN_ACTIVATION_RELU,
        CUDNN_ACTIVATION_TANH,
        CUDNN_ACTIVATION_CLIPPED_RELU

    cudnnStatus_t cudnnCreateActivationDescriptor(
        cudnnActivationDescriptor_t * activationDesc)

    cudnnStatus_t cudnnSetActivationDescriptor(
        cudnnActivationDescriptor_t         activationDesc,
        cudnnActivationMode_t               mode,
        cudnnNanPropagation_t               reluNanOpt,
        double                              reluCeiling)

    cudnnStatus_t cudnnGetActivationDescriptor(
        const cudnnActivationDescriptor_t   activationDesc,
        cudnnActivationMode_t * mode,
        cudnnNanPropagation_t * reluNanOpt,
        double * reluCeiling)

    cudnnStatus_t cudnnDestroyActivationDescriptor(
        cudnnActivationDescriptor_t activationDesc)

    cudnnStatus_t cudnnActivationForward(
        cudnnHandle_t                       handle,
        cudnnActivationDescriptor_t         activationDesc,
        const void * alpha,
        const cudnnTensorDescriptor_t       xDesc,
        const void * x,
        const void * beta,
        const cudnnTensorDescriptor_t       yDesc,
        void * y)

    cudnnStatus_t cudnnActivationBackward(
        cudnnHandle_t                       handle,
        cudnnActivationDescriptor_t         activationDesc,
        const void * alpha,
        const cudnnTensorDescriptor_t       yDesc,
        const void * y,
        const cudnnTensorDescriptor_t       dyDesc,
        const void * dy,
        const cudnnTensorDescriptor_t       xDesc,
        const void * x,
        const void * beta,
        const cudnnTensorDescriptor_t       dxDesc,
        void * dx)

    cudnnStatus_t cudnnCreateLRNDescriptor(
        cudnnLRNDescriptor_t * normDesc)

    # define CUDNN_LRN_MIN_N     1       # minimum allowed lrnN
    # define CUDNN_LRN_MAX_N     16      # maximum allowed lrnN
    # define CUDNN_LRN_MIN_K     1e-5    # minimum allowed lrnK
    # define CUDNN_LRN_MIN_BETA  0.01    # minimum allowed lrnBeta

    ctypedef enum cudnnLRNMode_t:
        # Normalize across tensor's dimA[1] dimension
        CUDNN_LRN_CROSS_CHANNEL_DIM1 = 0,

    cudnnStatus_t cudnnSetLRNDescriptor(
        cudnnLRNDescriptor_t                normDesc,
        unsigned                            lrnN,
        double                              lrnAlpha,
        double                              lrnBeta,
        double                              lrnK)
    cudnnStatus_t cudnnGetLRNDescriptor(
        cudnnLRNDescriptor_t                normDesc,
        unsigned * lrnN,
        double * lrnAlpha,
        double * lrnBeta,
        double * lrnK)

    cudnnStatus_t cudnnDestroyLRNDescriptor(cudnnLRNDescriptor_t lrnDesc)

    cudnnStatus_t cudnnLRNCrossChannelForward(
        cudnnHandle_t                       handle,
        cudnnLRNDescriptor_t                normDesc,
        cudnnLRNMode_t                      lrnMode,
        const void * alpha,
        const cudnnTensorDescriptor_t       xDesc,
        const void * x,
        const void * beta,
        const cudnnTensorDescriptor_t       yDesc,
        void * y)

    cudnnStatus_t cudnnLRNCrossChannelBackward(
        cudnnHandle_t                       handle,
        cudnnLRNDescriptor_t                normDesc,
        cudnnLRNMode_t                      lrnMode,
        const void * alpha,
        const cudnnTensorDescriptor_t       yDesc,
        const void * y,
        const cudnnTensorDescriptor_t       dyDesc,
        const void * dy,
        const cudnnTensorDescriptor_t       xDesc,
        const void * x,
        const void * beta,
        const cudnnTensorDescriptor_t       dxDesc,
        void * dx)

    ctypedef enum cudnnDivNormMode_t:
        CUDNN_DIVNORM_PRECOMPUTED_MEANS = 0,

    cudnnStatus_t cudnnDivisiveNormalizationForward(
        cudnnHandle_t                       handle,
        cudnnLRNDescriptor_t                normDesc,
        cudnnDivNormMode_t                  mode,
        const void * alpha,
        const cudnnTensorDescriptor_t       xDesc,  # same desc for means, temp, temp2
        const void * x,
        const void * means,  # if NULL, means are assumed to be zero
        void * temp,
        void * temp2,
        const void * beta,
        const cudnnTensorDescriptor_t       yDesc,
        void * y)

    cudnnStatus_t cudnnDivisiveNormalizationBackward(
        cudnnHandle_t                       handle,
        cudnnLRNDescriptor_t                normDesc,
        cudnnDivNormMode_t                  mode,
        const void * alpha,
        # same desc for x, means, dy, temp, temp2
        const cudnnTensorDescriptor_t       xDesc,
        const void * x,
        const void * means,  # if NULL, means are assumed to be zero
        const void * dy,
        void * temp,
        void * temp2,
        const void * beta,
        const cudnnTensorDescriptor_t       dXdMeansDesc,  # same desc for dx, dMeans
        void * dx,  # output x differential
        void * dMeans)  # output means differential, can be NULL

    ctypedef enum cudnnBatchNormMode_t:
        # bnScale, bnBias tensor dims are 1xCxHxWx.. (one value per
        # CHW...-slice, normalized over N slice)
        CUDNN_BATCHNORM_PER_ACTIVATION,

        # bnScale, bnBias tensor dims are 1xCx1x1 (one value per C-dim
        # normalized over Nx1xHxW subtensors)
        CUDNN_BATCHNORM_SPATIAL,

    # define CUDNN_BN_MIN_EPSILON 1e-5 # Minimum epsilon allowed to be used in
    # the Batch Normalization formula

    cudnnStatus_t cudnnDeriveBNTensorDescriptor(
        cudnnTensorDescriptor_t             derivedBnDesc,
        const cudnnTensorDescriptor_t       xDesc,
        cudnnBatchNormMode_t                mode)

    cudnnStatus_t cudnnBatchNormalizationForwardTraining(
        cudnnHandle_t                       handle,
        cudnnBatchNormMode_t                mode,

        const void * alpha,  # alpha[0] = result blend factor
        const void * beta,  # beta[0] = dest layer blend factor

        const cudnnTensorDescriptor_t       xDesc,
        const void * x,     # NxCxHxW
        const cudnnTensorDescriptor_t       yDesc,
        void * y,     # NxCxHxW

        const cudnnTensorDescriptor_t       bnScaleBiasMeanVarDesc,

        # 'Gamma' and 'Beta' respectively in Ioffe and Szegedy's paper's notation
        const void * bnScale,
        const void * bnBias,

        double                              exponentialAverageFactor,

        void * resultRunningMean,
        void * resultRunningVariance,
        double                              epsilon,
        void * resultSaveMean,
        void * resultSaveInvVariance)

    cudnnStatus_t cudnnBatchNormalizationForwardInference(
        cudnnHandle_t                       handle,
        cudnnBatchNormMode_t                mode,
        const void * alpha,  # alpha[0] = result blend factor
        const void * beta,  # beta[0] = dest layer blend factor
        const cudnnTensorDescriptor_t       xDesc,
        const void * x,     # NxCxHxW
        const cudnnTensorDescriptor_t       yDesc,
        void * y,     # NxCxHxW
        const cudnnTensorDescriptor_t       bnScaleBiasMeanVarDesc,
        const void * bnScale,
        const void * bnBias,
        const void * estimatedMean,
        const void * estimatedVariance,
        double                              epsilon)

    cudnnStatus_t cudnnBatchNormalizationBackward(
        cudnnHandle_t                       handle,
        cudnnBatchNormMode_t                mode,
        const void * alphaDataDiff,
        const void * betaDataDiff,
        const void * alphaParamDiff,
        const void * betaParamDiff,
        const cudnnTensorDescriptor_t       xDesc,  # same desc for x, dx, dy
        const void * x,
        const cudnnTensorDescriptor_t       dyDesc,
        const void * dy,
        const cudnnTensorDescriptor_t       dxDesc,
        void * dx,
        const cudnnTensorDescriptor_t       dBnScaleBiasDesc,
        const void * bnScale,  # bnBias doesn't affect backpropagation
        void * dBnScaleResult,
        void * dBnBiasResult,
        double                              epsilon,
        const void * savedMean,
        const void * savedInvVariance)

    ctypedef enum cudnnSamplerType_t:
        CUDNN_SAMPLER_BILINEAR = 0,

    cudnnStatus_t cudnnCreateSpatialTransformerDescriptor(

        cudnnSpatialTransformerDescriptor_t * stDesc)

    cudnnStatus_t cudnnSetSpatialTransformerNdDescriptor(
        cudnnSpatialTransformerDescriptor_t         stDesc,
        cudnnSamplerType_t                          samplerType,
        cudnnDataType_t                             dataType,
        const int                                   nbDims,
        const int                                   dimA[])

    cudnnStatus_t cudnnDestroySpatialTransformerDescriptor(
        cudnnSpatialTransformerDescriptor_t        stDesc)

    cudnnStatus_t cudnnSpatialTfGridGeneratorForward(
        cudnnHandle_t                              handle,
        const cudnnSpatialTransformerDescriptor_t  stDesc,
        const void * theta,
        void * grid)

    cudnnStatus_t cudnnSpatialTfGridGeneratorBackward(
        cudnnHandle_t                              handle,
        const cudnnSpatialTransformerDescriptor_t  stDesc,
        const void * dgrid,
        void * dtheta)

    cudnnStatus_t cudnnSpatialTfSamplerForward(
        cudnnHandle_t                              handle,
        cudnnSpatialTransformerDescriptor_t        stDesc,
        const void * alpha,
        const cudnnTensorDescriptor_t              xDesc,
        const void * x,
        const void * grid,
        const void * beta,
        cudnnTensorDescriptor_t                    yDesc,
        void * y)

    cudnnStatus_t cudnnSpatialTfSamplerBackward(
        cudnnHandle_t                              handle,
        cudnnSpatialTransformerDescriptor_t        stDesc,
        const void * alpha,
        const cudnnTensorDescriptor_t              xDesc,
        const void * x,
        const void * beta,
        const cudnnTensorDescriptor_t              dxDesc,
        void * dx,
        const void * alphaDgrid,
        const cudnnTensorDescriptor_t              dyDesc,
        const void * dy,
        const void * grid,
        const void * betaDgrid,
        void * dgrid)

    ctypedef unsigned int cudnnDropoutDescriptor_t

    cudnnStatus_t cudnnCreateDropoutDescriptor(cudnnDropoutDescriptor_t * dropoutDesc)

    cudnnStatus_t cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc)

    cudnnStatus_t cudnnDropoutGetStatesSize(cudnnHandle_t handle, size_t * sizeInBytes)

    cudnnStatus_t cudnnDropoutGetReserveSpaceSize(cudnnTensorDescriptor_t xdesc, size_t * sizeInBytes)

    cudnnStatus_t cudnnSetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc,
                                            cudnnHandle_t handle,
                                            float dropout,
                                            void * states,
                                            size_t stateSizeInBytes,
                                            unsigned long long seed)

    cudnnStatus_t cudnnDropoutForward(cudnnHandle_t handle,
                                      const cudnnDropoutDescriptor_t dropoutDesc,
                                      const cudnnTensorDescriptor_t xdesc,
                                      const void * x,
                                      const cudnnTensorDescriptor_t ydesc,
                                      void * y,
                                      void * reserveSpace,
                                      size_t reserveSpaceSizeInBytes)

    cudnnStatus_t cudnnDropoutBackward(cudnnHandle_t handle,
                                       const cudnnDropoutDescriptor_t dropoutDesc,
                                       const cudnnTensorDescriptor_t dydesc,
                                       const void * dy,
                                       const cudnnTensorDescriptor_t dxdesc,
                                       void * dx,
                                       void * reserveSpace,
                                       size_t reserveSpaceSizeInBytes)

    ctypedef enum cudnnRNNMode_t:
        CUDNN_RNN_RELU,  # Stock RNN with ReLu activation
        CUDNN_RNN_TANH,  # Stock RNN with tanh activation
        CUDNN_LSTM,     # LSTM with no peephole connections
        # Using h' = tanh(r * Uh(t-1) + Wx) and h = (1 - z) * h' + z * h(t-1);
        CUDNN_GRU

    ctypedef enum cudnnDirectionMode_t:
        CUDNN_UNIDIRECTIONAL = 0,
        # Using output concatination at each step. Do we also want to support
        # output sum?
        CUDNN_BIDIRECTIONAL = 1

    ctypedef enum cudnnRNNInputMode_t:
        CUDNN_LINEAR_INPUT,
        CUDNN_SKIP_INPUT

    ctypedef unsigned int cudnnRNNDescriptor_t

    cudnnStatus_t cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t * rnnDesc)
    cudnnStatus_t cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t rnnDesc)

    cudnnStatus_t cudnnSetRNNDescriptor(cudnnRNNDescriptor_t rnnDesc,
                                        int hiddenSize,
                                        int numLayers,
                                        # Between layers, not between recurrent
                                        # steps.
                                        cudnnDropoutDescriptor_t dropoutDesc,
                                        cudnnRNNInputMode_t inputMode,
                                        cudnnDirectionMode_t direction,
                                        cudnnRNNMode_t mode,
                                        cudnnDataType_t dataType)

    # dataType in the RNN descriptor is used to determine math precision
    # dataType in weight descriptors and input descriptors is used to describe
    # storage

    cudnnStatus_t cudnnGetRNNWorkspaceSize(cudnnHandle_t              handle,
                                           const cudnnRNNDescriptor_t rnnDesc,
                                           const int seqLength,
                                           const cudnnTensorDescriptor_t * xDesc,
                                           size_t * sizeInBytes
                                           )

    cudnnStatus_t cudnnGetRNNTrainingReserveSize(cudnnHandle_t              handle,
                                                 const cudnnRNNDescriptor_t rnnDesc,
                                                 const int seqLength,
                                                 const cudnnTensorDescriptor_t * xDesc,
                                                 size_t * sizeInBytes
                                                 )

    cudnnStatus_t cudnnGetRNNParamsSize(cudnnHandle_t              handle,
                                        const cudnnRNNDescriptor_t rnnDesc,
                                        const cudnnTensorDescriptor_t    xDesc,
                                        size_t * sizeInBytes,
                                        cudnnDataType_t dataType
                                        )

    cudnnStatus_t cudnnGetRNNLinLayerMatrixParams(cudnnHandle_t              handle,
                                                  const cudnnRNNDescriptor_t rnnDesc,
                                                  const int layer,
                                                  const cudnnTensorDescriptor_t xDesc,
                                                  const cudnnFilterDescriptor_t wDesc,
                                                  const void * w,
                                                  const int linLayerID,
                                                  cudnnFilterDescriptor_t linLayerMatDesc,
                                                  void ** linLayerMat
                                                  )

    cudnnStatus_t cudnnGetRNNLinLayerBiasParams(cudnnHandle_t              handle,
                                                const cudnnRNNDescriptor_t rnnDesc,
                                                const int layer,
                                                const cudnnTensorDescriptor_t xDesc,
                                                const cudnnFilterDescriptor_t wDesc,
                                                const void * w,
                                                const int linLayerID,
                                                cudnnFilterDescriptor_t linLayerBiasDesc,
                                                void ** linLayerBias
                                                )

    cudnnStatus_t cudnnRNNForwardInference(cudnnHandle_t handle,
                                           const cudnnRNNDescriptor_t rnnDesc,
                                           const int seqLength,
                                           const cudnnTensorDescriptor_t * xDesc,
                                           const void * x,
                                           const cudnnTensorDescriptor_t hxDesc,
                                           const void * hx,
                                           const cudnnTensorDescriptor_t cxDesc,
                                           const void * cx,
                                           const cudnnFilterDescriptor_t wDesc,
                                           const void * w,
                                           const cudnnTensorDescriptor_t * yDesc,
                                           void * y,
                                           const cudnnTensorDescriptor_t hyDesc,
                                           void * hy,
                                           const cudnnTensorDescriptor_t cyDesc,
                                           void * cy,
                                           void * workspace,
                                           size_t workSpaceSizeInBytes)

    cudnnStatus_t cudnnRNNForwardTraining(cudnnHandle_t handle,
                                          const cudnnRNNDescriptor_t rnnDesc,
                                          const int seqLength,
                                          const cudnnTensorDescriptor_t * xDesc,
                                          const void * x,
                                          const cudnnTensorDescriptor_t hxDesc,
                                          const void * hx,
                                          const cudnnTensorDescriptor_t cxDesc,
                                          const void * cx,
                                          const cudnnFilterDescriptor_t wDesc,
                                          const void * w,
                                          const cudnnTensorDescriptor_t * yDesc,
                                          void * y,
                                          const cudnnTensorDescriptor_t hyDesc,
                                          void * hy,
                                          const cudnnTensorDescriptor_t cyDesc,
                                          void * cy,
                                          void * workspace,
                                          size_t workSpaceSizeInBytes,
                                          void * reserveSpace,
                                          size_t reserveSpaceSizeInBytes)

    cudnnStatus_t cudnnRNNBackwardData(cudnnHandle_t handle,
                                       const cudnnRNNDescriptor_t rnnDesc,
                                       const int seqLength,
                                       const cudnnTensorDescriptor_t * yDesc,
                                       const void * y,
                                       const cudnnTensorDescriptor_t * dyDesc,
                                       const void * dy,
                                       const cudnnTensorDescriptor_t dhyDesc,
                                       const void * dhy,
                                       const cudnnTensorDescriptor_t dcyDesc,
                                       const void * dcy,
                                       const cudnnFilterDescriptor_t wDesc,
                                       const void * w,
                                       const cudnnTensorDescriptor_t hxDesc,
                                       const void * hx,
                                       const cudnnTensorDescriptor_t cxDesc,
                                       const void * cx,
                                       const cudnnTensorDescriptor_t * dxDesc,
                                       void * dx,
                                       const cudnnTensorDescriptor_t dhxDesc,
                                       void * dhx,
                                       const cudnnTensorDescriptor_t dcxDesc,
                                       void * dcx,
                                       void * workspace,
                                       size_t workSpaceSizeInBytes,
                                       const void * reserveSpace,
                                       size_t reserveSpaceSizeInBytes)

    cudnnStatus_t cudnnRNNBackwardWeights(cudnnHandle_t handle,
                                          const cudnnRNNDescriptor_t rnnDesc,
                                          const int seqLength,
                                          const cudnnTensorDescriptor_t * xDesc,
                                          const void * x,
                                          const cudnnTensorDescriptor_t hxDesc,
                                          const void * hx,
                                          const cudnnTensorDescriptor_t * yDesc,
                                          const void * y,
                                          const void * workspace,
                                          size_t workSpaceSizeInBytes,
                                          const cudnnFilterDescriptor_t dwDesc,
                                          void * dw,
                                          const void * reserveSpace,
                                          size_t reserveSpaceSizeInBytes)

    cudnnStatus_t cudnnSetFilter4dDescriptor_v3(
        cudnnFilterDescriptor_t             filterDesc,
        cudnnDataType_t                     dataType,  # image data type
        int                                 k,        # number of output feature maps
        int                                 c,        # number of input feature maps
        int                                 h,        # height of each input filter
        int                                 w)      # width of  each input filter

    cudnnStatus_t cudnnSetFilter4dDescriptor_v4(
        cudnnFilterDescriptor_t             filterDesc,
        cudnnDataType_t                     dataType,  # image data type
        cudnnTensorFormat_t                 format,
        int                                 k,        # number of output feature maps
        int                                 c,        # number of input feature maps
        int                                 h,        # height of each input filter
        int                                 w)      # width of  each input filter

    cudnnStatus_t cudnnGetFilter4dDescriptor_v3(
        const cudnnFilterDescriptor_t       filterDesc,
        cudnnDataType_t * dataType,  # image data type
        int * k,        # number of output feature maps
        int * c,        # number of input feature maps
        int * h,        # height of each input filter
        int * w)      # width of  each input filter

    cudnnStatus_t cudnnGetFilter4dDescriptor_v4(
        const cudnnFilterDescriptor_t       filterDesc,
        cudnnDataType_t * dataType,  # image data type
        cudnnTensorFormat_t * format,
        int * k,        # number of output feature maps
        int * c,        # number of input feature maps
        int * h,        # height of each input filter
        int * w)      # width of  each input filter

    cudnnStatus_t cudnnSetFilterNdDescriptor_v3(
        cudnnFilterDescriptor_t             filterDesc,
        cudnnDataType_t                     dataType,  # image data type
        int                                 nbDims,
        const int                           filterDimA[])

    cudnnStatus_t cudnnSetFilterNdDescriptor_v4(
        cudnnFilterDescriptor_t             filterDesc,
        cudnnDataType_t                     dataType,  # image data type
        cudnnTensorFormat_t                 format,
        int                                 nbDims,
        const int                           filterDimA[])

    cudnnStatus_t cudnnGetFilterNdDescriptor_v3(
        const cudnnFilterDescriptor_t       filterDesc,
        int                                 nbDimsRequested,
        cudnnDataType_t * dataType,  # image data type
        int * nbDims,
        int                                 filterDimA[])

    cudnnStatus_t cudnnGetFilterNdDescriptor_v4(
        const cudnnFilterDescriptor_t       filterDesc,
        int                                 nbDimsRequested,
        cudnnDataType_t * dataType,  # image data type
        cudnnTensorFormat_t * format,
        int * nbDims,
        int                                 filterDimA[])

    cudnnStatus_t cudnnSetPooling2dDescriptor_v3(
        cudnnPoolingDescriptor_t            poolingDesc,
        cudnnPoolingMode_t                  mode,
        int                                 windowHeight,
        int                                 windowWidth,
        int                                 verticalPadding,
        int                                 horizontalPadding,
        int                                 verticalStride,
        int                                 horizontalStride)

    cudnnStatus_t cudnnSetPooling2dDescriptor_v4(
        cudnnPoolingDescriptor_t            poolingDesc,
        cudnnPoolingMode_t                  mode,
        cudnnNanPropagation_t               maxpoolingNanOpt,
        int                                 windowHeight,
        int                                 windowWidth,
        int                                 verticalPadding,
        int                                 horizontalPadding,
        int                                 verticalStride,
        int                                 horizontalStride)
    cudnnStatus_t cudnnGetPooling2dDescriptor_v3(
        const cudnnPoolingDescriptor_t      poolingDesc,
        cudnnPoolingMode_t * mode,
        int * windowHeight,
        int * windowWidth,
        int * verticalPadding,
        int * horizontalPadding,
        int * verticalStride,
        int * horizontalStride)

    cudnnStatus_t cudnnGetPooling2dDescriptor_v4(
        const cudnnPoolingDescriptor_t      poolingDesc,
        cudnnPoolingMode_t * mode,
        cudnnNanPropagation_t * maxpoolingNanOpt,
        int * windowHeight,
        int * windowWidth,
        int * verticalPadding,
        int * horizontalPadding,
        int * verticalStride,
        int * horizontalStride)

    cudnnStatus_t cudnnSetPoolingNdDescriptor_v3(
        cudnnPoolingDescriptor_t            poolingDesc,
        const cudnnPoolingMode_t            mode,
        int                                 nbDims,
        const int                           windowDimA[],
        const int                           paddingA[],
        const int                           strideA[])

    cudnnStatus_t cudnnSetPoolingNdDescriptor_v4(
        cudnnPoolingDescriptor_t            poolingDesc,
        const cudnnPoolingMode_t            mode,
        const cudnnNanPropagation_t         maxpoolingNanOpt,
        int                                 nbDims,
        const int                           windowDimA[],
        const int                           paddingA[],
        const int                           strideA[])

    cudnnStatus_t cudnnGetPoolingNdDescriptor_v3(
        const cudnnPoolingDescriptor_t      poolingDesc,
        const int                           nbDimsRequested,
        cudnnPoolingMode_t * mode,
        int * nbDims,
        int                                 windowDimA[],
        int                                 paddingA[],
        int                                 strideA[])

    cudnnStatus_t cudnnGetPoolingNdDescriptor_v4(
        const cudnnPoolingDescriptor_t      poolingDesc,
        int                                 nbDimsRequested,
        cudnnPoolingMode_t * mode,
        cudnnNanPropagation_t * maxpoolingNanOpt,
        int * nbDims,
        int                                 windowDimA[],
        int                                 paddingA[],
        int                                 strideA[])

    cudnnStatus_t cudnnActivationForward_v3(
        cudnnHandle_t                       handle,
        cudnnActivationMode_t               mode,
        const void * alpha,
        const cudnnTensorDescriptor_t       xDesc,
        const void * x,
        const void * beta,
        const cudnnTensorDescriptor_t       yDesc,
        void * y)

    cudnnStatus_t cudnnActivationForward_v4(
        cudnnHandle_t                       handle,
        cudnnActivationDescriptor_t         activationDesc,
        const void * alpha,
        const cudnnTensorDescriptor_t       xDesc,
        const void * x,
        const void * beta,
        const cudnnTensorDescriptor_t       yDesc,
        void * y)

    cudnnStatus_t cudnnActivationBackward_v3(
        cudnnHandle_t                       handle,
        cudnnActivationMode_t               mode,
        const void * alpha,
        const cudnnTensorDescriptor_t       yDesc,
        const void * y,
        const cudnnTensorDescriptor_t       dyDesc,
        const void * dy,
        const cudnnTensorDescriptor_t       xDesc,
        const void * x,
        const void * beta,
        const cudnnTensorDescriptor_t       dxDesc,
        void * dx)

    cudnnStatus_t cudnnActivationBackward_v4(
        cudnnHandle_t                       handle,
        cudnnActivationDescriptor_t         activationDesc,
        const void * alpha,
        const cudnnTensorDescriptor_t       yDesc,
        const void * y,
        const cudnnTensorDescriptor_t       dyDesc,
        const void * dy,
        const cudnnTensorDescriptor_t       xDesc,
        const void * x,
        const void * beta,
        const cudnnTensorDescriptor_t       dxDesc,
        void * dx)
