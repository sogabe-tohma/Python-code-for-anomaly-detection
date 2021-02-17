from renom.cuda.base.cuda_base cimport *

cdef extern from "curand.h":
    ctypedef enum curandStatus:
        # No errors
        CURAND_STATUS_SUCCESS = 0,
        # Header file and linked library version do not match
        CURAND_STATUS_VERSION_MISMATCH = 100,
        # Generator not initialized
        CURAND_STATUS_NOT_INITIALIZED = 101,
        # Memory allocation failed
        CURAND_STATUS_ALLOCATION_FAILED = 102,
        # Generator is wrong type
        CURAND_STATUS_TYPE_ERROR = 103,
        # Argument out of range
        CURAND_STATUS_OUT_OF_RANGE = 104,
        # Length requested is not a multple of dimension
        CURAND_STATUS_LENGTH_NOT_MULTIPLE = 105,
        # GPU does not have double precision required by MRG32k3a
        CURAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106,
        # Kernel launch failure
        CURAND_STATUS_LAUNCH_FAILURE = 201,
        # Preexisting failure on library entry
        CURAND_STATUS_PREEXISTING_FAILURE = 202,
        # Initialization of CUDA failed
        CURAND_STATUS_INITIALIZATION_FAILED = 203,
        # Architecture mismatch, GPU does not support requested feature
        CURAND_STATUS_ARCH_MISMATCH = 204,
        # Internal library error
        CURAND_STATUS_INTERNAL_ERROR = 999

    ctypedef curandStatus curandStatus_t

    ctypedef enum curandRngType:
        CURAND_RNG_TEST = 0,
        # Default pseudorandom generator
        CURAND_RNG_PSEUDO_DEFAULT = 100,
        # XORWOW pseudorandom generator
        CURAND_RNG_PSEUDO_XORWOW = 101,
        # MRG32k3a pseudorandom generator
        CURAND_RNG_PSEUDO_MRG32K3A = 121,
        # Mersenne Twister MTGP32 pseudorandom generator
        CURAND_RNG_PSEUDO_MTGP32 = 141,
        # Mersenne Twister MT19937 pseudorandom generator
        CURAND_RNG_PSEUDO_MT19937 = 142,
        # PHILOX-4x32-10 pseudorandom generator
        CURAND_RNG_PSEUDO_PHILOX4_32_10 = 161,
        # Default quasirandom generator
        CURAND_RNG_QUASI_DEFAULT = 200,
        # Sobol32 quasirandom generator
        CURAND_RNG_QUASI_SOBOL32 = 201,
        # Scrambled Sobol32 quasirandom generator
        CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 202,
        # Sobol64 quasirandom generator
        CURAND_RNG_QUASI_SOBOL64 = 203,
        # Scrambled Sobol64 quasirandom generator
        CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 204

    ctypedef curandRngType curandRngType_t

    ctypedef enum curandOrdering:
        # Best ordering for pseudorandom results
        CURAND_ORDERING_PSEUDO_BEST = 100,
        # Specific default 4096 thread sequence for pseudorandom results
        CURAND_ORDERING_PSEUDO_DEFAULT = 101,
        # Specific seeding pattern for fast lower quality pseudorandom results
        CURAND_ORDERING_PSEUDO_SEEDED = 102,
        # Specific n-dimensional ordering for quasirandom results
        CURAND_ORDERING_QUASI_DEFAULT = 201

    ctypedef curandOrdering curandOrdering_t

    ctypedef enum curandDirectionVectorSet:
        # Specific set of 32-bit direction vectors generated from polynomials
        # recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions
        CURAND_DIRECTION_VECTORS_32_JOEKUO6 = 101,
        # Specific set of 32-bit direction vectors generated from polynomials
        # recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions, and scrambled
        CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 = 102,
        # Specific set of 64-bit direction vectors generated from polynomials
        # recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions
        CURAND_DIRECTION_VECTORS_64_JOEKUO6 = 103,
        # Specific set of 64-bit direction vectors generated from polynomials
        # recommended by S. Joe and F. Y. Kuo, for up to 20,000 dimensions, and scrambled
        CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 = 104

    ctypedef struct curandDistribution_st
    ctypedef struct curandDistributionShift_st
    ctypedef struct curandDistributionM2Shift_st
    ctypedef struct curandHistogramM2_st
    ctypedef struct curandDiscreteDistribution_st

    ctypedef curandDirectionVectorSet curandDirectionVectorSet_t
    ctypedef unsigned int curandDirectionVectors32_t[32]
    ctypedef unsigned long long curandDirectionVectors64_t[64]
    ctypedef struct curandGenerator_st
    ctypedef curandGenerator_st *curandGenerator_t
    ctypedef double curandDistribution_st
    ctypedef curandDistribution_st *curandDistribution_t
    ctypedef curandDistributionShift_st *curandDistributionShift_t
    ctypedef curandDistributionM2Shift_st *curandDistributionM2Shift_t
    ctypedef curandHistogramM2_st *curandHistogramM2_t
    ctypedef unsigned int curandHistogramM2K_st
    ctypedef curandHistogramM2K_st *curandHistogramM2K_t
    ctypedef curandDistribution_st curandHistogramM2V_st
    ctypedef curandHistogramM2V_st *curandHistogramM2V_t

    ctypedef curandDiscreteDistribution_st *curandDiscreteDistribution_t
    ctypedef enum curandMethod:
        CURAND_CHOOSE_BEST = 0, # choose best depends on args
        CURAND_ITR = 1,
        CURAND_KNUTH = 2,
        CURAND_HITR = 3,
        CURAND_M1 = 4,
        CURAND_M2 = 5,
        CURAND_BINARY_SEARCH = 6,
        CURAND_DISCRETE_GAUSS = 7,
        CURAND_REJECTION = 8,
        CURAND_DEVICE_API = 9,
        CURAND_FAST_REJECTION = 10,
        CURAND_3RD = 11,
        CURAND_DEFINITION = 12,
        CURAND_POISSON = 13

    ctypedef curandMethod curandMethod_t

    curandStatus_t curandCreateGenerator(
        curandGenerator_t *generator,
        curandRngType_t rng_type)

    curandStatus_t curandCreateGeneratorHost(
        curandGenerator_t *generator,
        curandRngType_t rng_type)

    curandStatus_t curandDestroyGenerator(curandGenerator_t generator)

    curandStatus_t curandGetVersion(int *version)

    # curandStatus_t curandGetProperty(libraryPropertyType type, int *value)

    curandStatus_t curandSetStream(curandGenerator_t generator, cudaStream_t stream)

    curandStatus_t curandSetPseudoRandomGeneratorSeed(
        curandGenerator_t generator,
        unsigned long long seed)

    curandStatus_t curandSetGeneratorOffset(curandGenerator_t generator, unsigned long long offset)

    curandStatus_t curandSetGeneratorOrdering(curandGenerator_t generator, curandOrdering_t order)

    curandStatus_t curandSetQuasiRandomGeneratorDimensions(
        curandGenerator_t generator,
        unsigned int num_dimensions)

    curandStatus_t curandGenerate(
        curandGenerator_t generator,
        unsigned int *outputPtr, size_t num)

    curandStatus_t curandGenerateLongLong(
        curandGenerator_t generator,
        unsigned long long *outputPtr,
        size_t num)

    curandStatus_t curandGenerateUniform(
        curandGenerator_t generator,
        float *outputPtr, size_t num)

    curandStatus_t curandGenerateUniformDouble(
        curandGenerator_t generator,
        double *outputPtr, size_t num)

    curandStatus_t curandGenerateNormal(
        curandGenerator_t generator,
        float *outputPtr,
        size_t n,
        float mean,
        float stddev)

    curandStatus_t curandGenerateNormalDouble(
        curandGenerator_t generator,
        double *outputPtr,
        size_t n,
        double mean,
        double stddev)

    curandStatus_t curandGenerateLogNormal(
        curandGenerator_t generator,
        float *outputPtr,
        size_t n,
        float mean,
        float stddev)

    curandStatus_t curandGenerateLogNormalDouble(
        curandGenerator_t generator,
        double *outputPtr,
        size_t n,
        double mean,
        double stddev)

    curandStatus_t curandCreatePoissonDistribution(
        double lmd,
        curandDiscreteDistribution_t *discrete_distribution)

    curandStatus_t curandDestroyDistribution(
        curandDiscreteDistribution_t discrete_distribution)

    curandStatus_t curandGeneratePoisson(
        curandGenerator_t generator,
        unsigned int *outputPtr,
        size_t n, double lmd)

    # just for internal usage
    curandStatus_t curandGeneratePoissonMethod(
        curandGenerator_t generator,
        unsigned int *outputPtr,
        size_t n,
        double lmd,
        curandMethod_t method)

    curandStatus_t curandGenerateBinomial(
        curandGenerator_t generator,
        unsigned int *outputPtr,
        size_t num,
        unsigned int n,
        double p)

    # just for internal usage
    curandStatus_t curandGenerateBinomialMethod(
        curandGenerator_t generator,
        unsigned int *outputPtr,
        size_t num,
        unsigned int n,
        double p,
        curandMethod_t method)

    curandStatus_t curandGenerateSeeds(curandGenerator_t generator)

    curandStatus_t curandGetDirectionVectors32(
        curandDirectionVectors32_t *vectors[],
        curandDirectionVectorSet_t set)

    curandStatus_t curandGetScrambleConstants32(unsigned int **constants)

    curandStatus_t curandGetDirectionVectors64(
        curandDirectionVectors64_t *vectors[],
        curandDirectionVectorSet_t set)

    curandStatus_t curandGetScrambleConstants64(unsigned long long **constants)
