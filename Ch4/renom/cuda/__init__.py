import numpy as np
import traceback
import contextlib
import warnings
try:
    from renom.cuda.base.cuda_base import *
    from renom.cuda.cublas.cublas import *
    from renom.cuda.thrust.thrust import *
    from renom.cuda.curand.curand import *
    from renom.cuda.cudnn.cudnn import *
    _has_cuda = True
except ImportError:
    gpu_allocator = None
    curand_generator = None
    _has_cuda = False

    @contextlib.contextmanager
    def use_device(device_id):
        yield

_cuda_is_active = False
_cuda_is_disabled = False


def set_cuda_active(activate=True):
    '''If True is given, cuda will be activated.

    Args:
        activate (bool): Activation flag.
    '''
    global _cuda_is_active
    if not has_cuda() and activate:
        warnings.warn("Couldn't find cuda modules.")
    _cuda_is_active = activate


def is_cuda_active():
    """Checks whether CUDA is activated.

    Returns:
        (bool): True if cuda is active.
    """
    return _cuda_is_active and has_cuda() and not _cuda_is_disabled


def has_cuda():
    """This method checks cuda libraries are available.

    Returns:
        (bool): True if cuda is correctly set.
    """
    return _has_cuda


@contextlib.contextmanager
def use_cuda(is_active=True):
    # save cuda state
    cur = _cuda_is_active
    set_cuda_active(is_active)
    try:
        yield None
    finally:
        # restore cuda state
        set_cuda_active(cur)


@contextlib.contextmanager
def disable_cuda(is_disabled=True):
    global _cuda_is_disabled
    # save cuda state
    cur = _cuda_is_disabled
    _cuda_is_disabled = is_disabled
    try:
        yield None
    finally:
        # restore cuda state
        _cuda_is_disabled = cur


_CuRandGens = {}


def _create_curand(seed=None):
    deviceid = cuGetDevice()
    if seed is None:
        seed = seed if seed else np.random.randint(4294967295, size=1)

    ret = CuRandGen(seed)
    _CuRandGens[deviceid] = ret
    return ret


def curand_generator(seed=None):
    deviceid = cuGetDevice()
    if deviceid not in _CuRandGens:
        _create_curand()

    gen = _CuRandGens[deviceid]
    if seed is not None:
        gen.set_seed(seed)
    return gen


def curand_set_seed(seed):
    """Set a seed to curand random number generator.
    The curand generator is used when cuda is activated.

    Args:
        seed(int): Seed.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> from renom.cuda import set_cuda_active, curand_set_seed
        >>> set_cuda_active(True)
        >>> a = rm.Variable(np.arange(4).reshape(2, 2))
        >>> curand_set_seed(1)
        >>> print(rm.dropout(a))
        [[ 0.  0.]
         [ 4.  0.]]
        >>> curand_set_seed(1)
        >>> print(rm.dropout(a)) # Same result will be returned.
        [[ 0.  0.]
         [ 4.  0.]]

    """
    deviceid = cuGetDevice()
    assert deviceid in _CuRandGens, "Curand not set"
    _CuRandGens[deviceid].set_seed(seed)


def release_mem_pool():
    """This function releases GPU memory pool.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> from renom.cuda import set_cuda_active, release_mem_pool
        >>> set_cuda_active(True)
        >>> a = rm.Variable(np.arange(4).reshape(2, 2))
        >>> a.to_gpu()
        >>> a = None
        >>> release_mem_pool() # The data of array `a` will be released.
    """
    if gpu_allocator:
        gpu_allocator.release_pool()
