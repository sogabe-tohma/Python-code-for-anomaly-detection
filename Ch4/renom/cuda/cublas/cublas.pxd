# distutils: language=c++
from libc.stdint cimport uintptr_t
from libc.stdlib cimport malloc, free
cdef extern from "cuda_runtime.h":
  ctypedef struct CUstream:
    pass
  ctypedef CUstream * cudaStream_t

cdef extern from "cublas_v2.h":
    ctypedef float cuComplex        # Not a collect type.
    ctypedef double cuDoubleComplex  # Not a collect type.
    ctypedef struct cublasContext
    ctypedef cublasContext * cublasHandle_t

    ctypedef enum cublasOperation_t:
        CUBLAS_OP_N=0,
        CUBLAS_OP_T=1,
        CUBLAS_OP_C=2


    ctypedef enum cublasStatus_t:
        CUBLAS_STATUS_SUCCESS = 0,
        CUBLAS_STATUS_NOT_INITIALIZED = 1,
        CUBLAS_STATUS_ALLOC_FAILED = 3,
        CUBLAS_STATUS_INVALID_VALUE = 7,
        CUBLAS_STATUS_ARCH_MISMATCH = 8,
        CUBLAS_STATUS_MAPPING_ERROR = 11,
        CUBLAS_STATUS_EXECUTION_FAILED = 13,
        CUBLAS_STATUS_INTERNAL_ERROR = 14,
        CUBLAS_STATUS_NOT_SUPPORTED = 15,
        CUBLAS_STATUS_LICENSE_ERROR = 16

    # ---------------- CUBLAS BLAS1 functions ----------------
    # NRM2
    cublasStatus_t cublasSnrm2(cublasHandle_t, int n, const float * x, int incx)
    cublasStatus_t cublasDnrm2(cublasHandle_t, int n, const double * x, int incx)
    cublasStatus_t cublasScnrm2(cublasHandle_t, int n, const cuComplex * x, int incx)
    cublasStatus_t cublasDznrm2(cublasHandle_t, int n, const cuDoubleComplex * x, int incx)

    #------------------------------------------------------------------------
    # DOT
    cublasStatus_t cublasSdot(cublasHandle_t, int n, const float * x, int incx, const float * y,
                     int incy)
    cublasStatus_t cublasDdot(cublasHandle_t, int n, const double * x, int incx, const double * y,
                      int incy)
    cublasStatus_t cublasCdotu(cublasHandle_t, int n, const cuComplex * x, int incx, const cuComplex * y,
                          int incy)
    cublasStatus_t cublasCdotc(cublasHandle_t, int n, const cuComplex * x, int incx, const cuComplex * y,
                          int incy)
    cublasStatus_t cublasZdotu(cublasHandle_t, int n, const cuDoubleComplex * x, int incx, const cuDoubleComplex * y,
                                int incy)
    cublasStatus_t cublasZdotc(cublasHandle_t, int n, const cuDoubleComplex * x, int incx, const cuDoubleComplex * y,
                                int incy)

    #------------------------------------------------------------------------
    # SCAL
    cublasStatus_t cublasSscal(cublasHandle_t handle,int n, float * alpha, float * x, int incx)
    cublasStatus_t cublasDscal(cublasHandle_t handle,int n, double * alpha, double * x, int incx)
    cublasStatus_t cublasCscal(cublasHandle_t handle,int n, cuComplex alpha, cuComplex * x, int incx)
    cublasStatus_t cublasZscal(cublasHandle_t handle,int n, cuDoubleComplex alpha, cuDoubleComplex * x, int incx)

    cublasStatus_t cublasCsscal(cublasHandle_t handle,int n, float alpha, cuComplex * x, int incx)
    cublasStatus_t cublasZdscal(cublasHandle_t handle,int n, double alpha, cuDoubleComplex * x, int incx)
    #------------------------------------------------------------------------

    # AXPY
    cublasStatus_t cublasSaxpy(cublasHandle_t, int n, float * alpha, const float * x, int incx,
                     float * y, int incy)
    cublasStatus_t cublasDaxpy(cublasHandle_t, int n, double * alpha, const double * x,
                     int incx, double * y, int incy)
    cublasStatus_t cublasCaxpy(cublasHandle_t, int n, cuComplex alpha, const cuComplex * x,
                     int incx, cuComplex * y, int incy)
    cublasStatus_t cublasZaxpy(cublasHandle_t, int n, cuDoubleComplex alpha, const cuDoubleComplex * x,
                     int incx, cuDoubleComplex * y, int incy)

    #------------------------------------------------------------------------
    # COPY
    cublasStatus_t cublasScopy(cublasHandle_t, int n, const float * x, int incx, float * y,
                     int incy)
    cublasStatus_t cublasDcopy(cublasHandle_t, int n, const double * x, int incx, double * y,
                     int incy)
    cublasStatus_t cublasCcopy(cublasHandle_t, int n, const cuComplex * x, int incx, cuComplex * y,
                     int incy)
    cublasStatus_t cublasZcopy(cublasHandle_t, int n, const cuDoubleComplex * x, int incx, cuDoubleComplex * y,
                     int incy)
    #------------------------------------------------------------------------
    # SWAP
    cublasStatus_t cublasSswap(cublasHandle_t, int n, float * x, int incx, float * y, int incy)
    cublasStatus_t cublasDswap(cublasHandle_t, int n, double * x, int incx, double * y, int incy)
    cublasStatus_t cublasCswap(cublasHandle_t, int n, cuComplex * x, int incx, cuComplex * y, int incy)
    cublasStatus_t cublasZswap(cublasHandle_t, int n, cuDoubleComplex * x, int incx, cuDoubleComplex * y, int incy)
    #------------------------------------------------------------------------
    # AMAX
    cublasStatus_t cublasIsamax(cublasHandle_t, int n, const float * x, int incx)
    cublasStatus_t cublasIdamax(cublasHandle_t, int n, const double * x, int incx)
    cublasStatus_t cublasIcamax(cublasHandle_t, int n, const cuComplex * x, int incx)
    cublasStatus_t cublasIzamax(cublasHandle_t, int n, const cuDoubleComplex * x, int incx)
    #------------------------------------------------------------------------
    # AMIN
    cublasStatus_t cublasIsamin(cublasHandle_t, int n, const float * x, int incx)
    cublasStatus_t cublasIdamin(cublasHandle_t, int n, const double * x, int incx)

    cublasStatus_t cublasIcamin(cublasHandle_t, int n, const cuComplex * x, int incx)
    cublasStatus_t cublasIzamin(cublasHandle_t, int n, const cuDoubleComplex * x, int incx)
    #------------------------------------------------------------------------
    # ASUM
    cublasStatus_t cublasSasum(cublasHandle_t, int n, const float * x, int incx)
    cublasStatus_t cublasDasum(cublasHandle_t, int n, const double * x, int incx)
    cublasStatus_t cublasScasum(cublasHandle_t, int n, const cuComplex * x, int incx)
    cublasStatus_t cublasDzasum(cublasHandle_t, int n, const cuDoubleComplex * x, int incx)
    #------------------------------------------------------------------------
    # ROT
    cublasStatus_t cublasSrot(cublasHandle_t, int n, float * x, int incx, float * y, int incy,
                    float sc, float ss)
    cublasStatus_t cublasDrot(cublasHandle_t, int n, double * x, int incx, double * y, int incy,
                    double sc, double ss)
    cublasStatus_t cublasCrot(cublasHandle_t, int n, cuComplex * x, int incx, cuComplex * y,
                    int incy, float c, cuComplex s)
    cublasStatus_t cublasZrot(cublasHandle_t, int n, cuDoubleComplex * x, int incx,
                    cuDoubleComplex * y, int incy, double sc,
                    cuDoubleComplex cs)
    cublasStatus_t cublasCsrot(cublasHandle_t, int n, cuComplex * x, int incx, cuComplex * y,
                     int incy, float c, float s)
    cublasStatus_t cublasZdrot(cublasHandle_t, int n, cuDoubleComplex * x, int incx,
                     cuDoubleComplex * y, int incy, double c, double s)
    #------------------------------------------------------------------------
    # ROTG
    cublasStatus_t cublasSrotg(cublasHandle_t, float * sa, float * sb, float * sc, float * ss)
    cublasStatus_t cublasDrotg(cublasHandle_t, double * sa, double * sb, double * sc, double * ss)
    cublasStatus_t cublasCrotg(cublasHandle_t, cuComplex * ca, cuComplex cb, float * sc,
                     cuComplex * cs)
    cublasStatus_t cublasZrotg(cublasHandle_t, cuDoubleComplex * ca, cuDoubleComplex cb, double * sc,
                     cuDoubleComplex * cs)
    #------------------------------------------------------------------------
    # ROTM
    cublasStatus_t cublasSrotm(cublasHandle_t, int n, float * x, int incx, float * y, int incy,
                     const float * sparam)
    cublasStatus_t cublasDrotm(cublasHandle_t, int n, double * x, int incx, double * y, int incy,
                     const double * sparam)
    #------------------------------------------------------------------------
    # ROTMG
    cublasStatus_t cublasSrotmg(cublasHandle_t, float * sd1, float * sd2, float * sx1,
                      const float * sy1, float * sparam)
    cublasStatus_t cublasDrotmg(cublasHandle_t, double * sd1, double * sd2, double * sx1,
                      const double * sy1, double * sparam)

    # --------------- CUBLAS BLAS2 functions  ----------------
    # GEMV
    cublasStatus_t cublasSgemv(cublasHandle_t, char trans, int m, int n, float alpha,
                     const float * A, int lda, const float * x, int incx,
                     float beta, float * y, int incy)
    cublasStatus_t cublasDgemv(cublasHandle_t, char trans, int m, int n, double alpha,
                     const double * A, int lda, const double * x, int incx,
                     double beta, double * y, int incy)
    cublasStatus_t cublasCgemv(cublasHandle_t, char trans, int m, int n, cuComplex alpha,
                     const cuComplex * A, int lda, const cuComplex * x, int incx,
                     cuComplex beta, cuComplex * y, int incy)
    cublasStatus_t cublasZgemv(cublasHandle_t, char trans, int m, int n, cuDoubleComplex alpha,
                     const cuDoubleComplex * A, int lda, const cuDoubleComplex * x, int incx,
                     cuDoubleComplex beta, cuDoubleComplex * y, int incy)
    #------------------------------------------------------------------------
    # GBMV
    cublasStatus_t cublasSgbmv(cublasHandle_t, char trans, int m, int n, int kl, int ku,
                     float alpha, const float * A, int lda,
                     const float * x, int incx, float beta, float * y,
                     int incy)
    cublasStatus_t cublasDgbmv(cublasHandle_t, char trans, int m, int n, int kl, int ku,
                     double alpha, const double * A, int lda,
                     const double * x, int incx, double beta, double * y,
                     int incy)
    cublasStatus_t cublasCgbmv(cublasHandle_t, char trans, int m, int n, int kl, int ku,
                     cuComplex alpha, const cuComplex * A, int lda,
                     const cuComplex * x, int incx, cuComplex beta, cuComplex * y,
                     int incy)
    cublasStatus_t cublasZgbmv(cublasHandle_t, char trans, int m, int n, int kl, int ku,
                     cuDoubleComplex alpha, const cuDoubleComplex * A, int lda,
                     const cuDoubleComplex * x, int incx, cuDoubleComplex beta, cuDoubleComplex * y,
                     int incy)
    #------------------------------------------------------------------------
    # TRMV
    cublasStatus_t cublasStrmv(cublasHandle_t, char uplo, char trans, char diag, int n,
                     const float * A, int lda, float * x, int incx)
    cublasStatus_t cublasDtrmv(cublasHandle_t, char uplo, char trans, char diag, int n,
                     const double * A, int lda, double * x, int incx)
    cublasStatus_t cublasCtrmv(cublasHandle_t, char uplo, char trans, char diag, int n,
                     const cuComplex * A, int lda, cuComplex * x, int incx)
    cublasStatus_t cublasZtrmv(cublasHandle_t, char uplo, char trans, char diag, int n,
                     const cuDoubleComplex * A, int lda, cuDoubleComplex * x, int incx)
    #------------------------------------------------------------------------
    # TBMV
    cublasStatus_t cublasStbmv(cublasHandle_t, char uplo, char trans, char diag, int n, int k,
                     const float * A, int lda, float * x, int incx)
    cublasStatus_t cublasDtbmv(cublasHandle_t, char uplo, char trans, char diag, int n, int k,
                     const double * A, int lda, double * x, int incx)
    cublasStatus_t cublasCtbmv(cublasHandle_t, char uplo, char trans, char diag, int n, int k,
                     const cuComplex * A, int lda, cuComplex * x, int incx)
    cublasStatus_t cublasZtbmv(cublasHandle_t, char uplo, char trans, char diag, int n, int k,
                     const cuDoubleComplex * A, int lda, cuDoubleComplex * x, int incx)
    #------------------------------------------------------------------------
    # TPMV
    cublasStatus_t cublasStpmv(cublasHandle_t, char uplo, char trans, char diag, int n, const float * AP, float * x, int incx)

    cublasStatus_t cublasDtpmv(cublasHandle_t, char uplo, char trans, char diag, int n, const double * AP, double * x, int incx)

    cublasStatus_t cublasCtpmv(cublasHandle_t, char uplo, char trans, char diag, int n, const cuComplex * AP, cuComplex * x, int incx)

    cublasStatus_t cublasZtpmv(cublasHandle_t, char uplo, char trans, char diag, int n, const cuDoubleComplex * AP, cuDoubleComplex * x, int incx)
    #------------------------------------------------------------------------
    # TRSV
    cublasStatus_t cublasStrsv(cublasHandle_t, char uplo, char trans, char diag, int n, const float * A, int lda, float * x, int incx)

    cublasStatus_t cublasDtrsv(cublasHandle_t, char uplo, char trans, char diag, int n, const double * A, int lda, double * x, int incx)

    cublasStatus_t cublasCtrsv(cublasHandle_t, char uplo, char trans, char diag, int n, const cuComplex * A, int lda, cuComplex * x, int incx)

    cublasStatus_t cublasZtrsv(cublasHandle_t, char uplo, char trans, char diag, int n, const cuDoubleComplex * A, int lda,
                     cuDoubleComplex * x, int incx)
    #------------------------------------------------------------------------
    # TPSV
    cublasStatus_t cublasStpsv(cublasHandle_t, char uplo, char trans, char diag, int n, const float * AP,
                     float * x, int incx)

    cublasStatus_t cublasDtpsv(cublasHandle_t, char uplo, char trans, char diag, int n, const double * AP, double * x, int incx)

    cublasStatus_t cublasCtpsv(cublasHandle_t, char uplo, char trans, char diag, int n, const cuComplex * AP, cuComplex * x, int incx)

    cublasStatus_t cublasZtpsv(cublasHandle_t, char uplo, char trans, char diag, int n, const cuDoubleComplex * AP,
                     cuDoubleComplex * x, int incx)
    #------------------------------------------------------------------------
    # TBSV
    cublasStatus_t cublasStbsv(cublasHandle_t, char uplo, char trans,
                     char diag, int n, int k, const float * A,
                     int lda, float * x, int incx)

    cublasStatus_t cublasDtbsv(cublasHandle_t, char uplo, char trans,
                     char diag, int n, int k, const double * A,
                     int lda, double * x, int incx)
    cublasStatus_t cublasCtbsv(cublasHandle_t, char uplo, char trans,
                     char diag, int n, int k, const cuComplex * A,
                     int lda, cuComplex * x, int incx)

    cublasStatus_t cublasZtbsv(cublasHandle_t, char uplo, char trans,
                     char diag, int n, int k, const cuDoubleComplex * A,
                     int lda, cuDoubleComplex * x, int incx)
    #------------------------------------------------------------------------
    # SYMV/HEMV
    cublasStatus_t cublasSsymv(cublasHandle_t, char uplo, int n, float alpha, const float * A,
                     int lda, const float * x, int incx, float beta,
                     float * y, int incy)
    cublasStatus_t cublasDsymv(cublasHandle_t, char uplo, int n, double alpha, const double * A,
                     int lda, const double * x, int incx, double beta,
                     double * y, int incy)
    cublasStatus_t cublasChemv(cublasHandle_t, char uplo, int n, cuComplex alpha, const cuComplex * A,
                     int lda, const cuComplex * x, int incx, cuComplex beta,
                     cuComplex * y, int incy)
    cublasStatus_t cublasZhemv(cublasHandle_t, char uplo, int n, cuDoubleComplex alpha, const cuDoubleComplex * A,
                     int lda, const cuDoubleComplex * x, int incx, cuDoubleComplex beta,
                     cuDoubleComplex * y, int incy)
    #------------------------------------------------------------------------
    # SBMV/HBMV
    cublasStatus_t cublasSsbmv(cublasHandle_t, char uplo, int n, int k, float alpha,
                     const float * A, int lda, const float * x, int incx,
                     float beta, float * y, int incy)
    cublasStatus_t cublasDsbmv(cublasHandle_t, char uplo, int n, int k, double alpha,
                     const double * A, int lda, const double * x, int incx,
                     double beta, double * y, int incy)
    cublasStatus_t cublasChbmv(cublasHandle_t, char uplo, int n, int k, cuComplex alpha,
                     const cuComplex * A, int lda, const cuComplex * x, int incx,
                     cuComplex beta, cuComplex * y, int incy)
    cublasStatus_t cublasZhbmv(cublasHandle_t, char uplo, int n, int k, cuDoubleComplex alpha,
                     const cuDoubleComplex * A, int lda, const cuDoubleComplex * x, int incx,
                     cuDoubleComplex beta, cuDoubleComplex * y, int incy)
    #------------------------------------------------------------------------
    # SPMV/HPMV
    cublasStatus_t cublasSspmv(cublasHandle_t, char uplo, int n, float alpha,
                     const float * AP, const float * x,
                     int incx, float beta, float * y, int incy)
    cublasStatus_t cublasDspmv(cublasHandle_t, char uplo, int n, double alpha,
                     const double * AP, const double * x,
                     int incx, double beta, double * y, int incy)
    cublasStatus_t cublasChpmv(cublasHandle_t, char uplo, int n, cuComplex alpha,
                     const cuComplex * AP, const cuComplex * x,
                     int incx, cuComplex beta, cuComplex * y, int incy)
    cublasStatus_t cublasZhpmv(cublasHandle_t, char uplo, int n, cuDoubleComplex alpha,
                     const cuDoubleComplex * AP, const cuDoubleComplex * x,
                     int incx, cuDoubleComplex beta, cuDoubleComplex * y, int incy)

    #------------------------------------------------------------------------
    # GER
    cublasStatus_t cublasSger(cublasHandle_t, int m, int n, float alpha, const float * x, int incx,
                    const float * y, int incy, float * A, int lda)
    cublasStatus_t cublasDger(cublasHandle_t, int m, int n, double alpha, const double * x, int incx,
                    const double * y, int incy, double * A, int lda)

    cublasStatus_t cublasCgeru(cublasHandle_t, int m, int n, cuComplex alpha, const cuComplex * x,
                     int incx, const cuComplex * y, int incy,
                     cuComplex * A, int lda)
    cublasStatus_t cublasCgerc(cublasHandle_t, int m, int n, cuComplex alpha, const cuComplex * x,
                     int incx, const cuComplex * y, int incy,
                     cuComplex * A, int lda)
    cublasStatus_t cublasZgeru(cublasHandle_t, int m, int n, cuDoubleComplex alpha, const cuDoubleComplex * x,
                     int incx, const cuDoubleComplex * y, int incy,
                     cuDoubleComplex * A, int lda)
    cublasStatus_t cublasZgerc(cublasHandle_t, int m, int n, cuDoubleComplex alpha, const cuDoubleComplex * x,
                     int incx, const cuDoubleComplex * y, int incy,
                     cuDoubleComplex * A, int lda)
    #------------------------------------------------------------------------
    # SYR/HER
    cublasStatus_t cublasSsyr(cublasHandle_t, char uplo, int n, float alpha, const float * x,
                    int incx, float * A, int lda)
    cublasStatus_t cublasDsyr(cublasHandle_t, char uplo, int n, double alpha, const double * x,
                    int incx, double * A, int lda)

    cublasStatus_t cublasCher(cublasHandle_t, char uplo, int n, float alpha,
                    const cuComplex * x, int incx, cuComplex * A, int lda)
    cublasStatus_t cublasZher(cublasHandle_t, char uplo, int n, double alpha,
                    const cuDoubleComplex * x, int incx, cuDoubleComplex * A, int lda)

    #------------------------------------------------------------------------
    # SPR/HPR
    cublasStatus_t cublasSspr(cublasHandle_t, char uplo, int n, float alpha, const float * x,
                    int incx, float * AP)
    cublasStatus_t cublasDspr(cublasHandle_t, char uplo, int n, double alpha, const double * x,
                    int incx, double * AP)
    cublasStatus_t cublasChpr(cublasHandle_t, char uplo, int n, float alpha, const cuComplex * x,
                    int incx, cuComplex * AP)
    cublasStatus_t cublasZhpr(cublasHandle_t, char uplo, int n, double alpha, const cuDoubleComplex * x,
                    int incx, cuDoubleComplex * AP)
    #------------------------------------------------------------------------
    # SYR2/HER2
    cublasStatus_t cublasSsyr2(cublasHandle_t, char uplo, int n, float alpha, const float * x,
                     int incx, const float * y, int incy, float * A,
                     int lda)
    cublasStatus_t cublasDsyr2(cublasHandle_t, char uplo, int n, double alpha, const double * x,
                     int incx, const double * y, int incy, double * A,
                     int lda)
    cublasStatus_t cublasCher2(cublasHandle_t, char uplo, int n, cuComplex alpha, const cuComplex * x,
                     int incx, const cuComplex * y, int incy, cuComplex * A,
                     int lda)
    cublasStatus_t cublasZher2(cublasHandle_t, char uplo, int n, cuDoubleComplex alpha, const cuDoubleComplex * x,
                     int incx, const cuDoubleComplex * y, int incy, cuDoubleComplex * A,
                     int lda)

    #------------------------------------------------------------------------
    # SPR2/HPR2
    cublasStatus_t cublasSspr2(cublasHandle_t, char uplo, int n, float alpha, const float * x,
                     int incx, const float * y, int incy, float * AP)
    cublasStatus_t cublasDspr2(cublasHandle_t, char uplo, int n, double alpha,
                     const double * x, int incx, const double * y,
                     int incy, double * AP)
    cublasStatus_t cublasChpr2(cublasHandle_t, char uplo, int n, cuComplex alpha,
                     const cuComplex * x, int incx, const cuComplex * y,
                     int incy, cuComplex * AP)
    cublasStatus_t cublasZhpr2(cublasHandle_t, char uplo, int n, cuDoubleComplex alpha,
                     const cuDoubleComplex * x, int incx, const cuDoubleComplex * y,
                     int incy, cuDoubleComplex * AP)
    # ------------------------BLAS3 Functions -------------------------------
    # GEMM
    cublasStatus_t cublasSgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                     float *alpha, const float * A, int lda,
                     const float * B, int ldb, float *beta, float * C,
                     int ldc)
    cublasStatus_t cublasDgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                     double *alpha, const double * A, int lda,
                     const double * B, int ldb, double *beta, double * C,
                     int ldc)
    cublasStatus_t cublasCgemm(cublasHandle_t, char transa, char transb, int m, int n, int k,
                     cuComplex alpha, const cuComplex * A, int lda,
                     const cuComplex * B, int ldb, cuComplex beta,
                     cuComplex * C, int ldc)
    cublasStatus_t cublasZgemm(cublasHandle_t, char transa, char transb, int m, int n,
                     int k, cuDoubleComplex alpha,
                     const cuDoubleComplex * A, int lda,
                     const cuDoubleComplex * B, int ldb,
                     cuDoubleComplex beta, cuDoubleComplex * C,
                     int ldc)
    # -------------------------------------------------------
    # SYRK
    cublasStatus_t cublasSsyrk(cublasHandle_t, char uplo, char trans, int n, int k, float alpha,
                     const float * A, int lda, float beta, float * C,
                     int ldc)
    cublasStatus_t cublasDsyrk(cublasHandle_t, char uplo, char trans, int n, int k,
                     double alpha, const double * A, int lda,
                     double beta, double * C, int ldc)

    cublasStatus_t cublasCsyrk(cublasHandle_t, char uplo, char trans, int n, int k,
                     cuComplex alpha, const cuComplex * A, int lda,
                     cuComplex beta, cuComplex * C, int ldc)
    cublasStatus_t cublasZsyrk(cublasHandle_t, char uplo, char trans, int n, int k,
                     cuDoubleComplex alpha,
                     const cuDoubleComplex * A, int lda,
                     cuDoubleComplex beta,
                     cuDoubleComplex * C, int ldc)
    # -------------------------------------------------------
    # HERK
    cublasStatus_t cublasCherk(cublasHandle_t, char uplo, char trans, int n, int k,
                     float alpha, const cuComplex * A, int lda,
                     float beta, cuComplex * C, int ldc)
    cublasStatus_t cublasZherk(cublasHandle_t, char uplo, char trans, int n, int k,
                     double alpha,
                     const cuDoubleComplex * A, int lda,
                     double beta,
                     cuDoubleComplex * C, int ldc)
    # -------------------------------------------------------
    # SYR2K
    cublasStatus_t cublasSsyr2k(cublasHandle_t, char uplo, char trans, int n, int k, float alpha,
                      const float * A, int lda, const float * B, int ldb,
                      float beta, float * C, int ldc)

    cublasStatus_t cublasDsyr2k(cublasHandle_t, char uplo, char trans, int n, int k,
                      double alpha, const double * A, int lda,
                      const double * B, int ldb, double beta,
                      double * C, int ldc)
    cublasStatus_t cublasCsyr2k(cublasHandle_t, char uplo, char trans, int n, int k,
                      cuComplex alpha, const cuComplex * A, int lda,
                      const cuComplex * B, int ldb, cuComplex beta,
                      cuComplex * C, int ldc)

    cublasStatus_t cublasZsyr2k(cublasHandle_t, char uplo, char trans, int n, int k,
                      cuDoubleComplex alpha, const cuDoubleComplex * A, int lda,
                      const cuDoubleComplex * B, int ldb, cuDoubleComplex beta,
                      cuDoubleComplex * C, int ldc)
    # -------------------------------------------------------
    # HER2K
    cublasStatus_t cublasCher2k(cublasHandle_t, char uplo, char trans, int n, int k,
                      cuComplex alpha, const cuComplex * A, int lda,
                      const cuComplex * B, int ldb, float beta,
                      cuComplex * C, int ldc)

    cublasStatus_t cublasZher2k(cublasHandle_t, char uplo, char trans, int n, int k,
                      cuDoubleComplex alpha, const cuDoubleComplex * A, int lda,
                      const cuDoubleComplex * B, int ldb, double beta,
                      cuDoubleComplex * C, int ldc)

    #------------------------------------------------------------------------
    # SYMM
    cublasStatus_t cublasSsymm(cublasHandle_t, char side, char uplo, int m, int n, float alpha,
                     const float * A, int lda, const float * B, int ldb,
                     float beta, float * C, int ldc)
    cublasStatus_t cublasDsymm(cublasHandle_t, char side, char uplo, int m, int n, double alpha,
                     const double * A, int lda, const double * B, int ldb,
                     double beta, double * C, int ldc)

    cublasStatus_t cublasCsymm(cublasHandle_t, char side, char uplo, int m, int n, cuComplex alpha,
                     const cuComplex * A, int lda, const cuComplex * B, int ldb,
                     cuComplex beta, cuComplex * C, int ldc)

    cublasStatus_t cublasZsymm(cublasHandle_t, char side, char uplo, int m, int n, cuDoubleComplex alpha,
                     const cuDoubleComplex * A, int lda, const cuDoubleComplex * B, int ldb,
                     cuDoubleComplex beta, cuDoubleComplex * C, int ldc)
    #------------------------------------------------------------------------
    # HEMM
    cublasStatus_t cublasChemm(cublasHandle_t, char side, char uplo, int m, int n,
                     cuComplex alpha, const cuComplex * A, int lda,
                     const cuComplex * B, int ldb, cuComplex beta,
                     cuComplex * C, int ldc)
    cublasStatus_t cublasZhemm(cublasHandle_t, char side, char uplo, int m, int n,
                     cuDoubleComplex alpha, const cuDoubleComplex * A, int lda,
                     const cuDoubleComplex * B, int ldb, cuDoubleComplex beta,
                     cuDoubleComplex * C, int ldc)

    #------------------------------------------------------------------------
    # TRSM
    cublasStatus_t cublasStrsm(cublasHandle_t, char side, char uplo, char transa, char diag,
                     int m, int n, float alpha, const float * A, int lda,
                     float * B, int ldb)

    cublasStatus_t cublasDtrsm(cublasHandle_t, char side, char uplo, char transa,
                     char diag, int m, int n, double alpha,
                     const double * A, int lda, double * B,
                     int ldb)

    cublasStatus_t cublasCtrsm(cublasHandle_t, char side, char uplo, char transa, char diag,
                     int m, int n, cuComplex alpha, const cuComplex * A,
                     int lda, cuComplex * B, int ldb)

    cublasStatus_t cublasZtrsm(cublasHandle_t, char side, char uplo, char transa,
                     char diag, int m, int n, cuDoubleComplex alpha,
                     const cuDoubleComplex * A, int lda,
                     cuDoubleComplex * B, int ldb)
    #------------------------------------------------------------------------
    # TRMM
    cublasStatus_t cublasStrmm(cublasHandle_t, char side, char uplo, char transa, char diag,
                     int m, int n, float alpha, const float * A, int lda,
                     float * B, int ldb)
    cublasStatus_t cublasDtrmm(cublasHandle_t, char side, char uplo, char transa,
                     char diag, int m, int n, double alpha,
                     const double * A, int lda, double * B,
                     int ldb)
    cublasStatus_t cublasCtrmm(cublasHandle_t, char side, char uplo, char transa, char diag,
                     int m, int n, cuComplex alpha, const cuComplex * A,
                     int lda, cuComplex * B, int ldb)
    cublasStatus_t cublasZtrmm(cublasHandle_t, char side, char uplo, char transa,
                     char diag, int m, int n, cuDoubleComplex alpha,
                     const cuDoubleComplex * A, int lda, cuDoubleComplex * B,
                     int ldb)

    #-------------------------------------------------------------
    # Blas Extensions
    #

    cublasStatus_t cublasCreate_v2( cublasHandle_t * handle)
    cublasStatus_t cublasCreate(cublasHandle_t * handle)
    cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t stream)
    cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t *stream)
    cublasStatus_t cublasDestroy ( cublasHandle_t handle);

    # ---------------- CUBLAS BLAS-like extension ----------------
    # GEAM
    cublasStatus_t cublasSgeam(cublasHandle_t handle,
                               cublasOperation_t transa,
                               cublasOperation_t transb,
                               int m,
                               int n,
                               const float * alpha,
                               const float * A,
                               int lda,
                               const float * beta,
                               const float * B,
                               int ldb,
                               float * C,
                               int ldc)

    cublasStatus_t cublasDgeam(cublasHandle_t handle,
                               cublasOperation_t transa,
                               cublasOperation_t transb,
                               int m,
                               int n,
                               const double * alpha,
                               const double * A,
                               int lda,
                               const double * beta,
                               const double * B,
                               int ldb,
                               double * C,
                               int ldc)

    cublasStatus_t cublasCgeam(cublasHandle_t handle,
                               cublasOperation_t transa,
                               cublasOperation_t transb,
                               int m,
                               int n,
                               const cuComplex * alpha,
                               const cuComplex * A,
                               int lda,
                               const cuComplex * beta,
                               const cuComplex * B,
                               int ldb,
                               cuComplex * C,
                               int ldc)

    cublasStatus_t cublasZgeam(cublasHandle_t handle,
                               cublasOperation_t transa,
                               cublasOperation_t transb,
                               int m,
                               int n,
                               const cuDoubleComplex * alpha,  # host or device pointer
                               const cuDoubleComplex * A,
                               int lda,
                               const cuDoubleComplex * beta,  # host or device pointer
                               const cuDoubleComplex * B,
                               int ldb,
                               cuDoubleComplex * C,
                               int ldc)


cpdef cublas_axpy(gpu_value1, gpu_value2)
