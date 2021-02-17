# -*- coding: utf-8 -*-
from __future__ import print_function, division
import collections
import weakref
import numpy as np
from numbers import Number
from renom import precision
import renom.debug_graph
import renom.cuda
if renom.cuda.has_cuda():
    from renom.cuda.base import cuda_base
    from renom.cuda.gpuvalue.gpuvalue import GPUValue


class GraphAttrs(object):

    def __init__(self):
        object.__setattr__(self, 'v__attrs', {})

    def clear(self):
        self.v__attrs.clear()

    def get_names(self):
        return self.v__attrs.keys()

    def get_attrs(self):
        return self.v__attrs.values()

    def __setattr__(self, name, value):
        self.v__attrs[name] = value

    def __getattr__(self, name):
        try:
            return self.v__attrs[name]
        except KeyError:
            raise AttributeError('%r has no attribute %r' % (self, name))

    def get(self, key, default=None):
        return self.v__attrs.get(key, default)


class Node(np.ndarray):
    '''This is the base class of all operation function.
    Node class inherits numpy ndarray class.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> vx = rm.Variable(np.random.rand(3, 2))
        >>> isinstance(vx, rm.Node)
        True
    '''

    _gpu = None
    attrs = None
    _model = None
    _auto_update = False
    _no_backward = False
    _args = ()

    SHOWMARK = False

    _node_hook = None

    @classmethod
    def set_hook(cls, hook):
        cls._node_hook = hook

    def __new__(cls, value):
        ret = cls._create_node(value)
        return ret

    @classmethod
    def _run_node_hook(cls, ret):
        if cls._node_hook:
            ret = cls._node_hook.leave_create(cls, ret)
        return ret

    @classmethod
    def _create_node(cls, value):
        if isinstance(value, np.ndarray):
            ret = value.astype(precision).view(cls)
        elif renom.cuda.has_cuda() and isinstance(value, GPUValue):
            ret = super(Node, cls).__new__(
                cls, shape=value.shape, dtype=value.dtype)
            ret._gpu = value

        elif isinstance(value, Number):
            ret = np.array(value, dtype=precision).view(cls)
        else:
            raise ValueError('Invalid Node value: %r' % value)

        assert ret.dtype == precision, (
            "Type miss matched. Required is {}, actual is {}".format(
                precision().dtype, ret.dtype))

        ret.attrs = GraphAttrs()
        if renom.debug_graph.GET_ACTIVE_NODE() is not None:
            renom.debug_graph.SET_NODE_DICT(id(ret), ret)

        ret = cls._run_node_hook(ret)

        return ret

    @classmethod
    def calc_value(cls, *args, **kwargs):
        if renom.cuda.is_cuda_active():
            value = cls._oper_gpu(*args, **kwargs)
        else:
            value = cls._oper_cpu(*args, **kwargs)
        return value

    def __init__(self, *args, **kwargs):
        self.setflags(write=False)
        self._args = []
        q = collections.deque([args])
        while q:
            a = q.pop()
            if isinstance(a, Node):
                self._args.append(a)
            elif isinstance(a, list) or isinstance(a, tuple):
                q.extend(a)
            elif isinstance(a, dict):
                q.extend(a.values())
        self._args.extend(a for a in kwargs.values() if isinstance(a, Node))

        self._reduce_graph()
        return

    @property
    def auto_update(self):
        if self._auto_update:
            if self._model:
                if not self._model.auto_update:
                    return False
        return self._auto_update

    @auto_update.setter
    def auto_update(self, value):
        raise Exception()

    @property
    def prevent_update(self):
        if self._model:
            if self._model._prevent_update:
                return True
        return False

    @prevent_update.setter
    def prevent_update(self, value):
        raise Exception()

    @property
    def device_id(self):
        if self._gpu:
            return self._gpu.device_id

        if self._model:
            return self._model._device_id

        return 0

    def set_model(self, model):
        self._model = model

    def get_gpu(self):
        if not self._gpu:
            self._gpu = GPUValue(self)
        return self._gpu

    def set_gpu(self, gpu):
        self.release_gpu()
        self._gpu = gpu

    def to_cpu(self):
        '''Send the data from GPU device to CPU.'''
        if self._gpu:
            self._gpu.to_cpu(self)

    def to_gpu(self):
        '''Send the data on CPU to GPU device.
        This method only available if cuda is activated otherwise this raises `ValueError`.

        Example:
            >>> import numpy as np
            >>> import renom as rm
            >>> from renom.cuda import set_cuda_active
            >>> set_cuda_active(True)
            >>> a = rm.Variable(np.arange(4).reshape(2, 2))
            >>> a.to_gpu()  # Sending array to gpu device.
        '''
        if self._gpu:
            self._gpu.to_gpu(self)
        else:
            self._gpu = GPUValue(self)

    def copy(self):
        """Returns a copy of itself.
        If node object does not have data on gpu,
        this returns ndarray.

        Returns:
            (Node, ndarray): Copy of node object.
        """
        if self._gpu:
            return self.__class__(self._gpu.copy())
        else:
            return np.ndarray.copy(self)

    def copy_from(self, other):
        assert self.shape == other.shape
        assert self.dtype == other.dtype
        if self._gpu:
            if other._gpu:
                self._gpu.copy_from(other._gpu)
                return

        if hasattr(self, "setflags"):
            writable = self.flags.writeable
            self.setflags(write=True)

        try:
            self[...] = other
        finally:
            if hasattr(self, "setflags"):
                self.setflags(write=writable)

    def as_ndarray(self):
        '''This method returns itself as ndarray object.'''
        self.to_cpu()
        if self._gpu:
            return self._gpu.new_array()
        if isinstance(self, Number):
            return np.array(self, dtype=precision)
        else:
            if not self.flags['C_CONTIGUOUS']:
                self = np.ascontiguousarray(self)
            ret = np.ndarray(shape=self.shape, dtype=self.dtype, buffer=self)
            ret.setflags(write=True)
            return np.array(ret)

    def release_gpu(self):
        '''This method releases array data on GPU.'''
        if self._gpu:
            self._gpu = None

    def _update_diff(self, context, dy, **kwargs):
        ready = context.add(self, dy)
        if ready:
            diff = context.get(self)
            self.backward(context, diff, **kwargs)

    def _get_graph(self):
        if self.attrs:
            return self.attrs.get_attrs()
        return []

    def _has_autoupdate(self):
        '''Check if the graph to witch this node belongs need to update.'''

        for v in self._get_graph():
            if isinstance(v, Node):
                if v.auto_update:
                    return True

                if any((o is not None) for o in v._get_graph()):
                    return True

    def _reduce_graph(self):
        if self.attrs:
            if not self._has_autoupdate():
                self._no_backward = True
                self.attrs.clear()
                self._args = []
        return False

    def detach_graph(self):
        '''This method destroys computational graph.'''

        for v in self._get_graph():
            if isinstance(v, Node):
                v.detach_graph()
        if self.attrs:
            self.attrs.clear()

        self._args = []

    def backward(self, context, dy, **kwargs):
        if self._no_backward:
            return

        if renom.cuda.is_cuda_active():
            if self._gpu:
                with cuda_base.use_device(self._gpu.device_id):
                    return self._backward_gpu(context, dy, **kwargs)
            else:
                return self._backward_gpu(context, dy, **kwargs)
        else:
            return self._backward_cpu(context, dy, **kwargs)

    def __neg__(self):
        assert False

    def __pos__(self):
        return renom.core.Pos(self)

    def __abs__(self):
        assert False

    def __invert__(self):
        assert False

    def __add__(self, other):
        '''This method is defined in basic_ops.py'''
        assert False

    def __radd__(self, other):
        '''This method is defined in basic_ops.py'''
        assert False

    def __iadd__(self, other):
        '''This method is defined in basic_ops.py'''
        assert False

    def __sub__(self, other):
        '''This method is defined in basic_ops.py'''
        assert False

    def __rsub__(self, other):
        assert False

    def __isub__(self, other):
        assert False

    def __mul__(self, other):
        '''This method is defined in basic_ops.py'''
        assert False

    def __rmul__(self, other):
        assert False

    def __imul__(self, other):
        assert False

    def __div__(self, other):
        assert False

    def __rdiv__(self, other):
        assert False

    def __idiv__(self, other):
        assert False

    def __floordiv__(self, other):
        assert False

    def __rfloordiv__(self, other):
        assert False

    def __ifloordiv__(self, other):
        assert False

    def __truediv__(self, other):
        assert False

    def __rtruediv__(self, other):
        assert False

    def __itruediv__(self, other):
        assert False

    def __mod__(self, other):
        assert False

    def __rmod__(self, other):
        assert False

    def __imod__(self, other):
        assert False

    def __divmod__(self, other):
        assert False

    def __rdivmod__(self, other):
        assert False

    def __pow__(self, other):
        assert False

    def __rpow__(self, other):
        assert False

    def __ipow__(self, other):
        assert False

    def __lshift__(self, other):
        assert False

    def __rlshift__(self, other):
        assert False

    def __ilshift__(self, other):
        assert False

    def __rshift__(self, other):
        assert False

    def __rrshift__(self, other):
        assert False

    def __irshift__(self, other):
        assert False

    def __and__(self, other):
        assert False

    def __rand__(self, other):
        assert False

    def __iand__(self, other):
        assert False

    def __xor__(self, other):
        assert False

    def __rxor__(self, other):
        assert False

    def __ixor__(self, other):
        assert False

    def __or__(self, other):
        assert False

    def __ror__(self, other):
        assert False

    def __ior__(self, other):
        assert False

    def __getitem__(self, index):
        '''This method is defined in basic_ops.py'''
        assert False

    def __setitem__(self, index, value):
        if self._gpu is not None:
            self._gpu[index] = value
        else:
            np.ndarray.__setitem__(self, index, value)

    def __getslice__(self, i, j):
        assert False

    def __lt__(self, other):
        self.to_cpu()
        if hasattr(other, 'to_cpu'):
            other.to_cpu()
        return np.ndarray.__lt__(self, other)

    def __le__(self, other):
        self.to_cpu()
        if hasattr(other, 'to_cpu'):
            other.to_cpu()
        return np.ndarray.__le__(self, other)

    def __eq__(self, other):
        self.to_cpu()
        if hasattr(other, 'to_cpu'):
            other.to_cpu()
        return np.ndarray.__eq__(self, other)

    def __ne__(self, other):
        self.to_cpu()
        if hasattr(other, 'to_cpu'):
            other.to_cpu()
        return np.ndarray.__ne__(self, other)

    def __ge__(self, other):
        self.to_cpu()
        if hasattr(other, 'to_cpu'):
            other.to_cpu()
        return np.ndarray.__ge__(self, other)

    def __gt__(self, other):
        self.to_cpu()
        if hasattr(other, 'to_cpu'):
            other.to_cpu()
        return np.ndarray.__gt__(self, other)

    def __not__(self):
        self.to_cpu()
        return np.ndarray.__not__(self)

    def __str__(self):
        self.to_cpu()
        return np.ndarray.__str__(self.as_ndarray())

    def __repr__(self):
        self.to_cpu()
        return np.ndarray.__repr__(self)

    def __float__(self):
        self.to_cpu()
        return np.ndarray.__float__(self)

    def __int__(self):
        self.to_cpu()
        return np.ndarray.__int__(self)

    def __complex__(self):
        self.to_cpu()
        return np.ndarray.__complex__(self)

    def __bool__(self):
        self.to_cpu()
        return np.ndarray.__bool__(self)

    def __index__(self):
        self.to_cpu()
        return np.ndarray.__index__(self)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # move gpu values of input arrays to cpu
        new_inputs = []
        for item in inputs:
            if isinstance(item, Node):
                item.to_cpu()
                item.release_gpu()
                new_inputs.append(item.view(np.ndarray))
            else:
                new_inputs.append(item)

        # move gpu values of output arrays to cpu
        outs = kwargs.get('out', None)
        if isinstance(outs, tuple):
            new_outs = []
            for item in outs:
                if isinstance(item, Node):
                    item.to_cpu()
                    item.release_gpu()
                    new_outs.append(item.view(np.ndarray))
                else:
                    new_outs.append(item)

            kwargs['out'] = tuple(new_outs)

        elif outs is not None:
            kwargs['out'] = outs.view(np.ndarray)
            outs.to_cpu()
            outs.release_gpu()

        ret = getattr(ufunc, method)(*new_inputs, **kwargs)
        return ret

    def reshape(self, *shape):
        '''This method is defined in basic_ops.py'''
        assert False


class Variable(Node):
    '''Variable class.

    The gradient of this object will be calculated.
    Variable object is created from ndarray object or Number object.

    Args:
        value (Variable,ndarray): Input array.
        auto_update (bool): Auto update flag.
        weight_decay (float): Weight decay rate

    Weight decay allows the user to choose if weight decay is to be used in any
    of their variables.
    If weight decay is not defined in the Variable (I.e. defaults to None),
    then no weight decay is performed.

    For convenience, one can define a variable with a weight decay of 0 and provide
    the weight decay argument when building the gradients to default all weights to the
    same λ for weight decay.

    Individually assigned weight decay takes precedence over this default value,
    allowing users to customize the weight decay in the network.

    In summary, weight decay updates according to the following table.

    +-----------+-----------+--------------+
    | Variable  |   Grad    |   Result     |
    +===========+===========+==============+
    | None      |   <Any>   |   No Update  |
    +-----------+-----------+--------------+
    | 0.3       |   <Any>   |   0.3        |
    +-----------+-----------+--------------+
    | 0         |   None/0  |   No Update  |
    +-----------+-----------+--------------+
    | 0         |   0.3     |   0.3        |
    +-----------+-----------+--------------+



    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> x = np.array([1. -1])
        >>> rm.Variable(x)
        Variable([ 1., -1.], dtype=float32)
    '''

    weight_decay = None

    def __new__(cls, value, auto_update=True, weight_decay=None):
        ret = super(Variable, cls).__new__(cls, value)
        ret._auto_update = auto_update
        ret.weight_decay = weight_decay
        return ret

    def backward(self, context, dy, **kwargs):
        pass
