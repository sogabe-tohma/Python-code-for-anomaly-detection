import weakref
from numbers import Number
import itertools
import collections
import cython
import numpy as np
import renom.debug_graph as debug

try:
    from renom.cuda import is_cuda_active, use_device
    from renom.cuda.thrust.thrust import *
    from renom.cuda.base.cuda_base import *
    from renom.cuda.base import cuda_base
    from renom.cuda.cublas import cublas
except ImportError:
    pass


def _select_device(device_id):
    cur = cuGetDevice()
    cuSetDevice(device_id)  # switch device
    return cur


def get_gpu(array):
    f = getattr(array, 'get_gpu', None)
    if f:
        return f()

    if isinstance(array, np.ndarray):
        return GPUValue(array=array)
    elif isinstance(array, Number):
        return array
    else:
        raise Exception("Gpu not supported data type.")


def calc_broadcast_shape(*args):
    # silly, but works
    if all([isinstance(s, (np.ndarray, Number, GPUValue)) for s in args]):
        arrays = [np.empty(getattr(s, 'shape', 1), dtype=np.bool) for s in args]
        return np.broadcast(*arrays).shape
    else:
        raise Exception("Not supported data type.")


class _AdvIndex:
    def __init__(self, index):
        if isinstance(index, (list, tuple)):
            index = np.array(index)

        isbool = index.dtype.name == 'bool'
        if isbool:
            if isinstance(index, GPUValue):
                index = index.new_array()

            elems = []
            for j, v in enumerate(index.reshape(-1)):
                if v:
                    elems.append(j)

            index = np.array(elems, dtype='int64')
        elif isinstance(index, np.ndarray):
            index = index.astype('int64')

        self.org_index = index
        if not isinstance(index, GPUValue):
            index = index.reshape(-1)
            index = GPUValue(index.astype('int64'), dtype='int64')

        if index.dtype.type is not np.int64:
            raise IndexError("Invalid index type: %r" % index.dtype)

        self.shape = index.shape
        self.index = index


def _parse_index(arr, indexes):
    if not isinstance(indexes, tuple):
        indexes = [indexes]
    else:
        indexes = list(indexes)

    ellipsis = None
    num_values = 0

    # calc number of slice or int
    for i, s in enumerate(indexes):
        if s is None:
            continue

        elif s is Ellipsis:
            if ellipsis is not None:
                assert 0
            ellipsis = i
            continue

        num_values += 1

    # expand Ellipsis or append slices at tail
    if num_values != len(arr.shape):
        if ellipsis is None:
            ellipsis = len(indexes)

        f, b = indexes[:ellipsis], indexes[ellipsis + 1:]
        rest = len(arr.shape) - num_values
        mid = [slice(0, arr.shape[i + ellipsis], 1) for i in range(rest)]
        indexes = f + mid + b

    if len([i for i in indexes if i is not None]) != len(arr.shape):
        raise IndexError()

    # build slices
    slices = []
    dest_shapes = []
    result_shapes = []

    for i, index in enumerate(indexes):
        shape = arr.shape[len(slices)]

        if isinstance(index, slice):
            start, stop, step = index.indices(shape)
            slices.append((start, stop, step))

            dest_shape = 0
            if step < 0:
                if stop < start:
                    dest_shape = (start - stop - 1) // (-step) + 1
            else:
                if start < stop:
                    dest_shape = (stop - start - 1) // step + 1

            dest_shapes.append(dest_shape)
            result_shapes.append(dest_shape)

        elif isinstance(index, int):
            if index < 0:
                index = index + shape

            if not (0 <= index < shape):
                raise IndexError()

            slices.append((index, index + 1, 1))
            dest_shapes.append(1)

        else:
            # None(newaxis)
            result_shapes.append(1)

    strides = calc_strides(arr.shape)
    dest_strides = calc_strides(arr.shape)

    return slices, strides, dest_strides, result_shapes, dest_shapes


def build_shapes(arr, indexes):
    strides = calc_strides(arr.shape)

    # If a list of slices, change to list of slices
    if isinstance(indexes, list):
        # python built-in function all does not work for cython?
        all_slices = True
        for elem in indexes:
            if not isinstance(elem, slice):
                all_slices = False
                break
        if all_slices:
            indexes = tuple(indexes)

    # make indexes a list
    if isinstance(indexes, bool):
        slices = [[0, s, 1, None, st, st] for s, st in zip(arr.shape, strides)]
        return slices, [1 if indexes else 0] + list(arr.shape), list(arr.shape)

    elif isinstance(indexes, list):
        # if indexes is in form of `[[1]]`, then unwrap the outer list.
        for elem in indexes:
            if isinstance(elem, (list, tuple, np.ndarray, GPUValue)):
                indexes = indexes[:]
                break
        else:
            indexes = [indexes]
    elif isinstance(indexes, tuple):
        indexes = list(indexes)
    else:
        indexes = [indexes]

    # check if boolean index with same shape
    if len(indexes) == 1:
        elem = indexes[0]
        if isinstance(elem, (list, tuple, np.ndarray, GPUValue)):
            if not isinstance(elem, (np.ndarray, GPUValue)):
                elem = np.array(elem)
            if elem.dtype.name == 'bool':
                if elem.shape == arr.shape:
                    idxes = _AdvIndex(elem).index
                    slices = [[0, 0, 0, idxes, 1, 1]]
                    return slices, [idxes.size], [idxes.size]

    ellipsis = None
    num_values = 0
    is_advanced = False

    # calc number of slice or index
    for i, s in enumerate(indexes):
        # check if advanced index or not
        if isinstance(s, (list, tuple, np.ndarray, GPUValue)):
            is_advanced = True

        elif s is None:
            continue

        elif s is Ellipsis:
            if ellipsis is not None:
                assert 0
            ellipsis = i
            continue

        num_values += 1

    # expand Ellipsis or append slices at tail
    if num_values != len(arr.shape):
        if ellipsis is None:
            ellipsis = len(indexes)

        f, b = indexes[:ellipsis], indexes[ellipsis + 1:]
        rest = len(arr.shape) - num_values
        mid = [slice(0, arr.shape[i + ellipsis], 1) for i in range(rest)]
        indexes = f + mid + b

    if len([i for i in indexes if i is not None]) != len(arr.shape):
        raise IndexError()

    src_shape = arr.shape
    adv_shape = []
    if is_advanced:
        # convert int index to the advanced index
        # note that 1 in the [1, []] is an advanced index
        for i, elem in enumerate(indexes[:]):
            if isinstance(elem, int):
                indexes[i] = _AdvIndex([elem])
            elif isinstance(elem, (list, tuple, np.ndarray, GPUValue)):
                indexes[i] = _AdvIndex(elem)

        # collect advanced indexes
        advs = []
        stds = []
        num_advs = 0
        all = zip(indexes, strides, src_shape)
        for k, g in itertools.groupby(all, key=lambda e: isinstance(e[0], _AdvIndex)):
            if k:
                num_advs += 1
                advs.extend(g)
            else:
                stds.extend(g)

        # check if The advanced indexes are all next to each other.
        is_split_adv = (num_advs >= 2)

        if is_split_adv:
            # move adv indexes at topmost
            indexes = ([ind for ind, stride, shape in advs] +
                       [ind for ind, stride, shape in stds])
            strides = ([stride for ind, stride, shape in advs] +
                       [stride for ind, stride, shape in stds])
            src_shape = ([shape for ind, stride, shape in advs] +
                         [shape for ind, stride, shape in stds])

        adv_shape = calc_broadcast_shape(*[adv.org_index for adv, stride, shape in advs])

    # build slices
    # (start, stop, step, adv_indexes, stride, dest_stride)
    slices = []
    result_shapes = []
    dest_shapes = []
    adv_result_shapes = adv_shape[:]
    adv_ldxsize = calc_int_prod(adv_shape)
    adv_positions = []
    reduce_dim = []

    n_idx = 0
    for index in indexes:
        shape = src_shape[n_idx]
        stride = strides[n_idx]

        if isinstance(index, slice):
            start, stop, step = index.indices(shape)

            dest_shape = 0
            if step < 0:
                if stop < start:
                    dest_shape = (start - stop - 1) // (-step) + 1
            else:
                if start < stop:
                    dest_shape = (stop - start - 1) // step + 1

            slices.append([start, stop, step, None, stride])
            dest_shapes.append(dest_shape)
            result_shapes.append(dest_shape)
            n_idx += 1

        elif isinstance(index, int):
            if index < 0:
                index = index + shape

            if not (0 <= index < shape):
                raise IndexError()

            slices.append([index, index + 1, 1, None, stride])
            dest_shapes.append(1)
            reduce_dim.append(len(dest_shapes) - 1)
            n_idx += 1

        elif index is None:
            # None(newaxis)
            result_shapes.append(1)

        else:  # should be sequence
            adv_positions.append(len(slices))
            maxidx = cu_reduce_max(index.index)
            if maxidx.new_array() >= shape:
                raise IndexError()

            assert index.index
            slices.append([0, 0, 0, index.index, stride])
            if adv_result_shapes:
                dest_shapes.append(adv_ldxsize)
                result_shapes.extend(adv_result_shapes)
                adv_result_shapes = None

            n_idx += 1

    dest_strides = calc_strides(dest_shapes)
    dest_shapes = [d for i, d in enumerate(dest_shapes) if i not in reduce_dim]

    adv_dest_stride = dest_strides[adv_positions[0]] if adv_positions else None

    j = 0
    # set dest_stride
    for i in range(len(slices)):
        s = slices[i]
        if s[3] is None:
            # slice
            s.append(dest_strides[j])
            j += 1
        else:
            # adv index
            s.append(adv_dest_stride)
            j = adv_positions[0] + 1

    return slices, result_shapes, dest_shapes


def _build_broadcast_mask(left, right):
    if len(right) > len(left):
        reminds = right[:-1 * len(left)]
        for r in reminds:
            if r != 1:
                raise ValueError("could not broadcast")
        right = right[-1 * len(left):]
    elif len(right) < len(left):
        right = (1,) * (len(left) - len(right)) + right

    mask = []
    for lft, rgt in zip(left, right):
        if lft != rgt:
            if rgt != 1:
                raise ValueError("could not broadcast")
            mask.append(0)
        else:
            mask.append(1)

    return mask, right


class GPUValue(object):
    def __init__(self, array=None, shape=None, ptr=None, dtype=None):
        self._ptr = None
        if not is_cuda_active():
            raise ValueError('Cuda is not active. '
                             'Use renom.cuda.set_cuda_active() to activate.')

        if shape is not None:
            self.shape = tuple(shape)
        else:
            self.shape = getattr(array, "shape", None) or ()

        if not dtype:
            self.dtype = np.dtype(precision)
        else:
            self.dtype = np.dtype(dtype)

        self.itemsize = self.dtype.itemsize
        self.size = (calc_int_prod(self.shape) if self.shape else 1)
        self.nbytes = self.size * self.itemsize

        self._ptr = ptr
        if array is not None:
            self.to_gpu(array)
        elif not self._ptr:
            self.alloc()
        else:
            self.device_id = cuGetDevice()

        if debug.GET_ACTIVE_GPU() is not None:
            debug.SET_GPU_DICT(id(self), self)

        assert self._ptr
        self._ptr.refcount += 1

    def __dealloc__(self):
        self._ptr.refcount -= 1
        if self._ptr.refcount is 0:
            cuda_base.c_gpu_allocator.free(self._ptr)

    # Del is not called for extension classes
    # def __del__(self):
    #    self._free()

    def alloc(self):
        self._free()

        self._ptr = cuda_base.get_gpu_allocator().malloc(self.nbytes)
        self.device_id = cuGetDevice()

        assert self._ptr

    def _free(self):
        if self._ptr:
            cuda_base.get_gpu_allocator().free(self._ptr)
        self._ptr = None

    def __len__(self):
        if len(self.shape) > 0:
            return self.shape[0]
        else:
            return 1

    def reshape(self, *shape):
        # TODO: Find a way to create shapes without requesting potentially large
        # blocks of  temporary CPU memory.
        a = np.empty(self.shape, dtype=np.bool).reshape(*shape)
        # TODO: Currently shape size is checked during numpy reshaping, but this results in
        # a numpy reshaping error when the issue occurs within GPUValue, should probably
        # be a GPUValue error
        # assert np.prod(a.shape) == np.prod(self.shape), 'Requested shape has size {0} but original GPUValue \
        # has size {1}'.format(np.prod(a.shape),np.prod(self.shape))
        ret = GPUValue(ptr=self._ptr, shape=a.shape)
        return ret

    def get_gpu(self):
        return self

    def copy(self):
        if cuGetDevice() == self.device_id:
            ret = GPUValue(shape=self.shape)
            self._ptr.memcpyD2D(ret._ptr, self.nbytes)
        else:
            with use_device(self.device_id):
                arr = self.new_array()
            ret = GPUValue(arr)
        return ret

    def empty_like_me(self):
        ret = GPUValue(shape=self.shape)
        return ret

    def zeros_like_me(self):
        ret = self.empty_like_me()
        cufill(0., ret)
        return ret

    def ones_like_me(self):
        ret = self.empty_like_me()
        cufill(1., ret)
        return ret

    def new_array(self):
        em = np.empty(self.shape, dtype=self.dtype)
        self._ptr.memcpyD2H(em, em.nbytes)
        return em

    def to_cpu(self, value):
        assert self._ptr
        assert tuple(value.shape) == tuple(self.shape), "{} {}".format(value.shape, self.shape)
        assert value.dtype == self.dtype
        self._ptr.memcpyD2H(value, value.nbytes)
        return value

    def to_gpu(self, value):
        if value.dtype is not self.dtype:
            value = value.astype(self.dtype)

        assert value.shape == self.shape, "{} {}".format(value.shape, self.shape)

        if not self._ptr:
            self.nbytes = value.nbytes
            self.alloc()

        # todo: value.flatten() copies buffer
        with use_device(self.device_id):
            self._ptr.memcpyH2D(value.ravel(), value.nbytes)

    def copy_from(self, other):
        self._ptr.copy_from(other._ptr, self.nbytes)

    def transpose(self, axis):
        return cu_transpose(self, axis)

    def split(self, indices_or_sections, axis=0):
        N = self.shape[axis]  # Raises IndexError if axis is invalid

        try:
            len(indices_or_sections)
        except TypeError:
            size, mod = divmod(N, indices_or_sections)
            if N % indices_or_sections:
                raise ValueError(
                    'array split does not result in an equal division')
            indices_or_sections = range(size, N, size)

        slices = []
        for s in self.shape:
            slices.append(slice(0, s, 1))

        ret = []
        pos = 0
        for to in indices_or_sections:
            slices[axis] = slice(pos, to, 1)
            v = self[tuple(slices)]
            ret.append(v)
            pos = to

        if to < N:
            slices[axis] = slice(to, N, 1)
            v = self[tuple(slices)]
            ret.append(v)

        return ret

    def hsplit(self, indices_or_sections):
        return self.split(indices_or_sections, 1)

    def __pos__(self):
        ret = self.empty_like_me()
        cumul(self, 1, ret)
        return ret

    def __neg__(self):
        ret = self.empty_like_me()
        cumul(self, -1, ret)
        return ret

    def __matmul__(self, other):
        return self.dot(other)

    def dot(self, other):
        new_shape = self.shape[0], other.shape[1]
        ret = GPUValue(shape=new_shape)
        cublas.cublas_gemm(self, 0, other, 0, ret)
        return ret

    def sigmoid(self):
        ret = self.empty_like_me()
        cusigmoid(self, ret)
        return ret

    def tanh(self):
        ret = self.empty_like_me()
        cutanh(self, ret)
        return ret

    def __add__(self, other):
        with use_device(self.device_id):
            new_shape = calc_broadcast_shape(self, other)
            ret = GPUValue(shape=new_shape)
            # Only data type float32 is acceptable.
            cuadd(self, other, ret)
            return ret

    def __iadd__(self, other):
        with use_device(self.device_id):
            assert getattr(self, "shape", (1,)) == getattr(self, "shape", (1,))
            cublas.cublas_axpy(get_gpu(other), get_gpu(self))
            return self

    def __mul__(self, other):
        if not isinstance(self, GPUValue):
            return other.__rmul__(self)

        with use_device(self.device_id):
            new_shape = calc_broadcast_shape(self, other)
            ret = GPUValue(shape=new_shape)
            cumul(self, other, ret)
            return ret

    def __rmul__(self, other):
        with use_device(self.device_id):
            return self.__mul__(other)

    def __div__(self, other):
        if not isinstance(self, GPUValue):
            return other.__rdiv__(self)

        with use_device(self.device_id):
            return self.__truediv__(other)

    def __rdiv__(self, other):
        with use_device(self.device_id):
            return self.__rtruediv__(other)

    def __idiv__(self, other):
        with use_device(self.device_id):
            return self.__itruediv__(other)

    def __truediv__(self, other):
        if not isinstance(self, GPUValue):
            return other.__rtruediv__(self)

        with use_device(self.device_id):
            new_shape = calc_broadcast_shape(self, other)
            ret = GPUValue(shape=new_shape)
            cudiv(self, other, ret)
            return ret

    def __rtruediv__(self, other):
        with use_device(self.device_id):
            new_shape = calc_broadcast_shape(self, other)
            ret = GPUValue(shape=new_shape)
            curdiv(self, other, ret)
            return ret

    def __itruediv__(self, other):
        with use_device(self.device_id):
            assert getattr(self, "shape", (1,)) == getattr(self, "shape", (1,))
            new_shape = calc_broadcast_shape(self, other)
            ret = GPUValue(shape=new_shape)
            cudiv(self, other, ret)
            return ret

    def __sub__(self, other):
        with use_device(self.device_id):
            new_shape = calc_broadcast_shape(self, other)
            ret = GPUValue(shape=new_shape)
            cusub(self, other, ret)
            return ret

    def __isub__(self, other):
        with use_device(self.device_id):
            assert getattr(self, "shape", (1,)) == getattr(self, "shape", (1,))
            cublas.cublas_axpy(-get_gpu(other), get_gpu(self))
            return self

    def _oper_pow(self, other):
        if not isinstance(self, GPUValue):
            return other.__rpow__(self, modulo)

        with use_device(self.device_id):
            new_shape = calc_broadcast_shape(self, other)
            ret = GPUValue(shape=new_shape)
            cupow(self, other, ret)
            return ret

    def __pow__(self, other, modulo):
        return self._oper_pow(other)

    if not cython.compiled:
        __pow__ = _oper_pow  # noqa

    def __rpow__(self, other, modulo):
        with use_device(self.device_id):
            new_shape = calc_broadcast_shape(self, other)
            ret = GPUValue(shape=new_shape)
            curpow(self, other, ret)
            return ret

    def __getitem__(self, indexes):
        with use_device(self.device_id):
            slices, result_shapes, dest_shapes = build_shapes(self, indexes)

            dest_size = calc_int_prod(dest_shapes)

            ret = cu_get_item(self, self.size, dest_size, slices)

            ret.shape = tuple(result_shapes)
            return ret

    def __setitem__(self, indexes, value):
        with use_device(self.device_id):
            value = get_gpu(value)
            slices, result_shapes, dest_shapes = build_shapes(self, indexes)
            if calc_int_prod(result_shapes) == 0:
                return

            dest_strides = calc_strides(dest_shapes)
            mask, broadcasted = _build_broadcast_mask(dest_shapes, value.shape)

            broadcasted_strides = calc_strides(broadcasted)
            broadcasted_strides = [m * b for m, b in zip(mask, broadcasted_strides)]

            valuesize = calc_int_prod(dest_shapes)

            cu_set_item(value, valuesize, self, slices, dest_strides, broadcasted_strides)

    @property
    def T(self):
        with use_device(self.device_id):
            n = len(self.shape)
            assert n < 3
            clone = self.zeros_like_me()
            if n == 2:
                new_shape = list(clone.shape)
                with cublas.cublas_handler() as cublas_handle:
                    cublas.cublas_transpose(cublas_handle, self, clone)
                new_shape[0] = clone.shape[1]
                new_shape[1] = clone.shape[0]
                clone.shape = tuple(new_shape)
            return clone


try:
    from graphviz import Digraph
except ImportError:
    def plot_graph(n):   # NOQA
        pass


ACTIVE_GPU = None
ACTIVE_NODE = None


def DEBUG_GRAPH_INIT(active):
    global ACTIVE_GPU, ACTIVE_NODE
    if active:
        ACTIVE_GPU = weakref.WeakValueDictionary()
        ACTIVE_NODE = weakref.WeakValueDictionary()
    else:
        ACTIVE_GPU = None
        ACTIVE_NODE = None


def DEBUG_GPU_STAT():
    if ACTIVE_GPU is None:
        return

    print('Num of GPUValue: %d' % len(ACTIVE_GPU))
    print('Bytes of GPU   : %d' % sum(g.nbytes for g in ACTIVE_GPU))


def DEBUG_GET_ROOTS():
    if ACTIVE_NODE is None:
        return []

    forwards = collections.defaultdict(set)
    for o in ACTIVE_NODE.values():
        for ref in o._args:
            forwards[id(ref)].add(id(o))
    rootids = set(ACTIVE_NODE.keys()) - set(forwards.keys())
    roots = [ACTIVE_NODE[o] for o in rootids]

    return roots


def DEBUG_NODE_STAT():
    if ACTIVE_NODE is None:
        return

    print('Num of Node: %d' % len(ACTIVE_NODE))

    print('')
    print('Num of Node by types:')

    c = collections.Counter(str(o.__class__) for o in ACTIVE_NODE.values())

    print('-----------------------------------------------------')
    print(' #\t class')
    print('-----------------------------------------------------')
    for name, n in c.most_common():
        print('%d \t%s' % (n, name))

    length = collections.Counter()

    def walk(o, n):
        if not isinstance(o, Node):
            length[n + 1] += 1
            return

        if not o.attrs:
            return
        attrs = o.attrs.get_attrs()
        if not attrs:
            length[n + 1] += 1
        else:
            for attr in attrs:
                walk(attr, n + 1)

    for root in DEBUG_GET_ROOTS():
        walk(root, 0)

    print('')
    print('Num of terminal node by graph length:')

    print('-----------------------------------------------------')
    print('#\t length')
    print('-----------------------------------------------------')
    for length, n in length.most_common():
        print('%d \t%s' % (n, length))


def DEBUG_NODE_GRAPH():
    if ACTIVE_NODE is None:
        return
    roots = DEBUG_GET_ROOTS()
    _plot_graph(roots)


def _plot_graph(objs):
    g = Digraph('G', filename='graphviz_output')
    s = set()
    for n in objs:
        g.node(str(id(n)), str(type(n)))
        s.add(id(n))

        def add_edge(node):
            if not hasattr(node, "attrs"):
                return

            nodeid = str(id(node))
            if not node.attrs:
                return
            for val in node._args:
                valid = str(id(val))
                name = ''
                g.node(valid, label=str(type(val)))
                g.edge(valid, nodeid, label=name)

            for o in node._args:
                if id(o) not in s:
                    add_edge(o)
                    s.add(id(o))

        add_edge(n)

    g.view()
