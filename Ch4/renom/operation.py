# -*- coding: utf-8 -*.t-
from __future__ import print_function, division

import numpy as np
from renom.core import Node, BinOp, UnaryOp, to_value, Reshape
from renom.debug_graph import showmark
from renom.config import precision

try:
    from renom.cuda import *
    from renom.cuda.cublas import *
    from renom.cuda.base.cuda_base import *
    from renom.cuda.gpuvalue import GPUValue, get_gpu
except ImportError:
    pass


class Abase(Node):

    def __new__(cls, arg, axis=None, keepdims=False):
        assert isinstance(axis, (type(None), int)), 'This function only accepts int or None.'
        value, index = cls.calc_value(arg, axis, keepdims)
        ret = super(Abase, cls).__new__(cls, value)
        ret.attrs._arg = arg
        ret.attrs._axis = axis
        ret.attrs._index = index
        ret.attrs._keepdims = keepdims
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            axis = self.attrs._axis
            index = self.attrs._index
            dx = np.zeros(self.attrs._arg.shape, dtype=dy.dtype)

            if axis is None:
                dxx = dx.reshape(-1)
                dxx[index] = dy
            else:
                axis_list = list(range(len(dx.shape)))
                axis_list.pop(axis)
                axis_list.append(axis)
                rev = [-1] * len(axis_list)
                for i, a in enumerate(axis_list):
                    rev[a] = i
                dxx = np.transpose(dx, axis_list)
                if(not self.attrs._keepdims):
                    dyy = dy
                else:
                    axis_list = list(range(len(dy.shape)))
                    axis_list.pop(axis)
                    axis_list.append(axis)
                    rev = [-1] * len(axis_list)
                    for i, a in enumerate(axis_list):
                        rev[a] = i
                    dyy = np.transpose(dy, axis_list)
                for i in np.ndindex(index.shape):
                    dxx[i][index[i]] = dyy[i]

            # dxx is a representation of the same memory as dx

            self.attrs._arg._update_diff(context, dx, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            axis = self.attrs._axis
            index = self.attrs._index.new_array()
            dx = np.zeros(self.attrs._arg.shape, dy.dtype)

            if axis is None:
                dxx = dx.reshape(-1)
                dxx[index] = dy
            else:
                axis_list = list(range(len(dx.shape)))
                axis_list.pop(axis)
                axis_list.append(axis)
                rev = [-1] * len(axis_list)
                for i, a in enumerate(axis_list):
                    rev[a] = i
                dxx = np.transpose(dx, axis_list)
                if(not self.attrs._keepdims):
                    dyy = dy
                else:
                    dyy = np.transpose(dy, axis_list)
                for i in np.ndindex(index.shape):
                    dxx[i][index[i]] = dyy[i]
            self.attrs._arg._update_diff(context, get_gpu(dx), **kwargs)


class Amax(Abase):
    """This function performs max calculation.

    Args:
        arg (Variable, ndarray): Input matrix.
        axis (int): Perform calculation along this argument.
        keepdims (bool): If `True` is passed, reduced dimensions remain.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> # Forward Calculation
        >>> a = np.arange(4).reshape(2, 2)
        >>> a
        [[0 1]
         [2 3]]
        >>> rm.amax(a, axis=1)
        [ 1.  3.]
        >>>
        >>> rm.amax(a, axis=0)
        [ 2.  3.]
        >>> rm.amax(a, axis=0, keepdims=True)
        [[ 2.  3.]]
        >>>
        >>> # Calculation of differentiation
        >>> va = rm.Variable(a)
        >>> out = rm.amax(va)
        >>> grad = out.grad()
        >>> grad.get(va) # Getting the gradient of 'va'.
        [[ 0.,  0.],
         [ 0.,  1.]]
    """

    @classmethod
    def _oper_cpu(cls, arg, axis, keepdims):
        array = to_value(arg)
        # Max is calculated twice, update?
        return np.amax(array, axis, keepdims=keepdims), np.argmax(array, axis)

    @classmethod
    def _oper_gpu(cls, arg, axis, keepdims):
        array = get_gpu(arg)
        value = cu_reduce_max(array, axis, keepdims)
        index = cu_reduce_argmax(array, axis)
        return value, index


class Amin(Abase):
    """This function performs min calculation.

    Args:
        arg (Variable, ndarray): Input matrix.
        axis (int): Perform calculation along this argument.
        keepdims (bool): If `Ture` is passed, reduced dimensions remain.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> # Forward Calculation
        >>> a = np.arange(4).reshape(2, 2)
        >>> a
        [[0 1]
         [2 3]]
        >>> rm.amin(a, axis=1)
        [ 0.  2.]
        >>>
        >>> rm.amin(a, axis=0)
        [ 0.  1.]
        >>> rm.amin(a, axis=0, keepdims=True)
        [[ 0.  1.]]
        >>>
        >>> # Calculation of differentiation
        >>> va = rm.Variable(a)
        >>> out = rm.amin(va)
        >>> grad = out.grad()
        >>> grad.get(va) # Getting the gradient of 'va'.
        [[ 1.,  0.],
         [ 0.,  0.]]
    """

    @classmethod
    def _oper_cpu(cls, arg, axis, keepdims):
        array = to_value(arg)
        return np.amin(array, axis, keepdims=keepdims), np.argmin(array, axis)

    @classmethod
    def _oper_gpu(cls, arg, axis, keepdims):
        array = get_gpu(arg)
        value = cu_reduce_min(array, axis, keepdims)
        index = cu_reduce_argmin(array, axis)
        return value, index


def reshape(array, shape):
    """This function reshapes array.

    Args:
        array (Node): Input array.
        shape (tuple): Shape.

    Returns:
        (Node): Reshaped array.

    Example:
        >>> import renom as rm
        >>> import numpy as np
        >>> x = rm.Variable(np.arange(6))
        >>> x.shape
        (6,)
        >>> y = rm.reshape(x, (2, 3))
        >>> y.shape
        (2, 3)
    """
    return Reshape(array, shape)


class sum(Node):
    '''
    This function sums up matrix elements.
    If the argument 'axis' is passed, this function performs
    sum along specified axis.

    Args:
        array (Node): Input array.
        axis (int): Summing up along this axis.
        keepdims (bool): If this is True, dimension will not be reduced.

    Returns:
        (Node): Summed array.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>>
        >>> x = np.random.rand(2, 3)
        >>> z = rm.sum(x)
        >>> z
        sum(3.21392822265625, dtype=float32)
    '''

    @classmethod
    def _oper_cpu(cls, arg, axis=None, keepdims=False):
        return np.sum(arg, axis=axis, keepdims=keepdims)

    @classmethod
    def _oper_gpu(cls, arg, axis=None, keepdims=False):
        return cusum(get_gpu(arg), axis=axis, keepdims=keepdims)

    def __new__(cls, arg, axis=None, keepdims=False):
        value = cls.calc_value(arg, axis, keepdims=keepdims)
        ret = super(sum, cls).__new__(cls, value)
        ret.attrs._axis = axis
        ret.attrs._arg = arg
        ret.attrs._keep = keepdims
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            arg = self.attrs._arg
            axis = self.attrs._axis
            if axis is None or axis == 0:
                dx = np.ones_like(arg) * dy
            else:
                if not self.attrs._keep:
                    if isinstance(axis, int):
                        expanded = np.expand_dims(dy, axis)
                    else:
                        expanded = dy
                        for ax in axis:
                            expanded = np.expand_dims(expanded, ax)
                    dx = np.ones_like(arg) * expanded
                else:
                    dx = np.ones_like(arg) * dy
            arg._update_diff(context, dx, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            arg = self.attrs._arg
            axis = self.attrs._axis
            if axis is None or axis == 0:
                dx = get_gpu(arg).ones_like_me() * get_gpu(dy)
            else:
                dy = get_gpu(dy).new_array()
                if not self.attrs._keep:
                    if isinstance(axis, int):
                        expanded = np.expand_dims(dy, axis)
                    else:
                        expanded = dy
                        for ax in axis:
                            expanded = np.expand_dims(expanded, ax)
                    dx = np.ones_like(arg, dtype=arg.dtype) * expanded
                else:
                    dx = np.ones_like(arg, dtype=arg.dtype) * dy
            arg._update_diff(context, get_gpu(dx), **kwargs)


class dot(BinOp):
    '''
    This function executes dot product of the two matrixes.

    Args:
        lhs (Node,ndarray): Input array.
        rhs (Node,ndarray): Input array.

    Returns:
        (Node): Multiplied array.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>>
        >>> x = np.random.rand(2, 3)
        >>> y = np.random.rand(2, 2)
        >>> z = rm.dot(y, x)
        >>> z
        dot([[ 0.10709135,  0.15022227,  0.12853521],
             [ 0.30557284,  0.32320538,  0.26753256]], dtype=float32)
    '''

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.dot(lhs, rhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        new_shape = (lhs.shape[0], rhs.shape[1])
        ret = GPUValue(shape=new_shape)
        cublas_gemm(get_gpu(lhs), 0,
                    get_gpu(rhs), 0,
                    get_gpu(ret))
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._lhs, Node):
            self.attrs._lhs._update_diff(context, np.dot(dy, to_value(self.attrs._rhs).T), **kwargs)

        if isinstance(self.attrs._rhs, Node):
            self.attrs._rhs._update_diff(context, np.dot(to_value(self.attrs._lhs).T, dy), **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        lhs = self.attrs._lhs
        rhs = self.attrs._rhs
        if isinstance(self.attrs._lhs, Node):
            new_shape = lhs.shape
            ldx = GPUValue(shape=new_shape)
            cublas_gemm(get_gpu(dy), 0,
                        get_gpu(rhs), 1,
                        get_gpu(ldx))
            self.attrs._lhs._update_diff(context, ldx, **kwargs)

        if isinstance(self.attrs._rhs, Node):
            new_shape = rhs.shape
            rdx = GPUValue(shape=new_shape)
            cublas_gemm(get_gpu(lhs), 1,
                        get_gpu(dy), 0,
                        get_gpu(rdx))
            self.attrs._rhs._update_diff(context, rdx, **kwargs)


def _matmul(a, b):
    return dot(a, b)


Node.__matmul__ = _matmul


@showmark
class concat(Node):
    """
    Join a sequence of arrays along specified axis.

    Args:
        args (Node, List of Node): Input arrays or tuple of input arrays.
        axis (int): Concatenation will be performed along this axis. Default value is 1.

    Returns:
        (Node): Concatenated array.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>>
        >>> x = np.random.rand(2, 3)
        >>> y = np.random.rand(2, 2)
        >>> z = rm.concat(x, y)
        >>> z.shape
        (2, 5)
        >>> z
        concat([[ 0.56989014,  0.50372809,  0.40573129,  0.17601326,  0.07233092],
                [ 0.09377897,  0.8510806 ,  0.78971916,  0.52481949,  0.06913455]], dtype=float32)

    """

    @classmethod
    def _oper_cpu(cls, args, axis):
        return np.concatenate(args, axis=axis).copy()

    @classmethod
    def _oper_gpu(cls, args, axis):
        newshape = args[0].shape[:axis] + \
            (np.sum([a.shape[axis] for a in args]), ) + args[0].shape[axis + 1:]

        ret = GPUValue(shape=newshape)
        cuconcat([get_gpu(a) for a in args], ret, axis)
        return ret

    def __new__(cls, *args, **kwargs):
        if isinstance(args[0], (tuple, list)):
            args = args[0]
        axis = kwargs.get('axis', 1)
        assert all([len(args[0].shape) == len(args[i].shape) for i in range(1, len(args))]), \
            "All arguments must have same number of dimension."
        assert np.sum(np.sum(np.array([list(map(lambda x, y: x != y, args[0].shape, args[i].shape))
                                       for i in range(1, len(args))]), axis=0).astype(np.bool)) < 2, \
            "All dimensions must have same size except specified axis."

        val = cls.calc_value(args, axis)
        ret = super(concat, cls).__new__(cls, val)
        tmp = 0
        index = []
        for a in args[:-1]:
            tmp += a.shape[axis]
            index.append(tmp)
        ret.attrs._index = index
        ret.attrs._axis = axis
        for i, v in enumerate(args):
            setattr(ret.attrs, "_arg%d" % i, args[i])
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        axis = self.attrs._axis
        args = np.split(to_value(dy), self.attrs._index, axis=axis)
        for i in range(len(self.attrs._index) + 1):
            arg = getattr(self.attrs, "_arg%d" % i)
            if isinstance(arg, Node):
                arg._update_diff(context, args[i], **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        axis = self.attrs._axis
        args = get_gpu(dy).split(self.attrs._index, axis=axis)
        for i in range(len(self.attrs._index) + 1):
            arg = getattr(self.attrs, "_arg%d" % i)
            if isinstance(arg, Node):
                arg._update_diff(context, args[i], **kwargs)


class where(Node):
    """
    Return elements, either from a or b, depending on condition.

    Args:
        condition (Node, ndarray): Condition array.
        a (Node, ndarray): Input array.
        b (Node, ndarray): Input array.

    Returns:
        (Node): Conditioned array.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>>
        >>> x = np.random.rand(2, 3)
        >>> x
        array([[ 0.56989017,  0.50372811,  0.4057313 ],
               [ 0.09377897,  0.85108059,  0.78971919]])
        >>> z = rm.where(x > 0.5, x, 0)
        >>> z
        where([[ 0.56989014,  0.50372809,  0.        ],
               [ 0.        ,  0.8510806 ,  0.78971916]], dtype=float32)

    """

    @classmethod
    def _oper_cpu(cls, condition, a, b):
        return np.where(condition, a, b)

    @classmethod
    def _oper_gpu(cls, condition, a, b):
        a_cpu = getattr(get_gpu(a), "new_array()", a)
        b_cpu = getattr(get_gpu(b), "new_array()", b)
        ret = GPUValue(np.where(condition, a_cpu, b_cpu))
        return ret

    def __new__(cls, condition, a, b):
        value = cls.calc_value(condition, a, b)
        ret = super(where, cls).__new__(cls, value)
        ret.attrs._condition = condition
        ret.attrs._a, ret.attrs._b = a, b
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._a, Node):
            ldy = np.zeros_like(self.attrs._a)
            ldy[self.attrs._condition] = dy[self.attrs._condition]
            self.attrs._a._update_diff(context, ldy, **kwargs)

        if isinstance(self.attrs._b, Node):
            rdy = np.zeros_like(self.attrs._b)
            rdy[- self.attrs._condition] = dy[- self.attrs._condition]
            self.attrs._b._update_diff(context, rdy, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._a, Node):
            ldy = get_gpu(self.attrs._a).zeros_like_me()
            ldy[self.attrs._condition] = dy[self.attrs._condition]
            self.attrs._a._update_diff(context, ldy, **kwargs)

        if isinstance(self.attrs._b, Node):
            rdy = get_gpu(self.attrs._b).zeros_like_me()
            rdy[- self.attrs._condition] = dy[- self.attrs._condition]
            self.attrs._b._update_diff(context, rdy, **kwargs)


class sqrt(UnaryOp):
    """
    Square root operation.

    Args:
        arg (Node,ndarray): Input array.

    Returns:
       (Node): Square root of input array.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>>
        >>> x = np.random.rand(2, 3)
        >>> x
        array([[ 0.56989017,  0.50372811,  0.4057313 ],
               [ 0.09377897,  0.85108059,  0.78971919]])
        >>> z = rm.sqrt(x)
        >>> z
        sqrt([[ 0.75491071,  0.70973808,  0.6369704 ],
              [ 0.30623353,  0.92254031,  0.88866144]], dtype=float32)
    """

    @classmethod
    def _oper_cpu(cls, arg):
        return np.sqrt(arg)

    @classmethod
    def _oper_gpu(cls, arg):
        ret = GPUValue(shape=arg.shape)
        cusqrt(get_gpu(arg), ret)
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            dx = 0.5 / self
            self.attrs._arg._update_diff(context, dx * dy, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            dx = 0.5 / self
            self.attrs._arg._update_diff(context, dx * dy, **kwargs)


class square(UnaryOp):
    """Square operation.

    Args:
        arg (Node, ndarray): Input array.

    Returns:
        (Node): Squared array.

    """

    @classmethod
    def _oper_cpu(cls, arg):
        return arg * arg

    @classmethod
    def _oper_gpu(cls, arg):
        ret = GPUValue(shape=arg.shape)
        cupow(get_gpu(arg), 2, ret)
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            dx = self.attrs._arg * 2
            self.attrs._arg._update_diff(context, dx * dy, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            dx = self.attrs._arg * 2
            self.attrs._arg._update_diff(context, dx * dy, **kwargs)


class log(UnaryOp):
    """
    Log operation.

    Args:
        arg (Node,ndarray): Input array.

    Returns:
        (Node): Logarithm of input array.
    """

    @classmethod
    def _oper_cpu(cls, arg):
        return np.log(arg)

    @classmethod
    def _oper_gpu(cls, arg):
        ret = get_gpu(arg).empty_like_me()
        culoge(get_gpu(arg), ret)
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            self.attrs._arg._update_diff(context, dy / self.attrs._arg, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            self.attrs._arg._update_diff(context, dy / get_gpu(self.attrs._arg), **kwargs)


class exp(UnaryOp):
    """
    Exponential operation.

    Args:
        arg (Node, ndarray): Input array.

    Returns:
        (Node): Exponential of input array.
    """

    @classmethod
    def _oper_cpu(cls, arg):
        return np.exp(arg)

    @classmethod
    def _oper_gpu(cls, arg):
        ret = get_gpu(arg).empty_like_me()
        cuexp(get_gpu(arg), ret)
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            self.attrs._arg._update_diff(context, dy * self, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            self.attrs._arg._update_diff(context, dy * get_gpu(self), **kwargs)


class amin(Amin):
    """Returns min value or array of given array.
    You can specify the axis which the operation will be performed for.

    Args:
        arg (Node, ndarray): Input matrix.
        axis (int): Perform calculation along this argument.
        keepdims (bool): If `Ture` is passed, dimensions will not be reduced.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> # Forward Calculation
        >>> a = np.arange(4).reshape(2, 2)
        >>> a
        [[0 1]
         [2 3]]
        >>> rm.amin(a, axis=1)
        [ 0.  2.]
        >>>
        >>> rm.amin(a, axis=0)
        [ 0.  1.]
        >>> rm.amin(a, axis=0, keepdims=True)
        [[ 0.  1.]]
        >>>
        >>> # Calculation of differentiation
        >>> va = rm.Variable(a)
        >>> out = rm.amin(va)
        >>> grad = out.grad()
        >>> grad.get(va) # Getting the gradient of 'va'.
        [[ 1.,  0.],
         [ 0.,  0.]]

    """
    pass


class amax(Amax):
    """Returns max value or array of given array.
    You can specify the axis which the operation will be performed for.

    Args:
        arg (Node, ndarray): Input matrix.
        axis (int): Perform calculation along this argument.
        keepdims (bool): If `Ture` is passed, dimensions will not be reduced.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> # Forward Calculation
        >>> a = np.arange(4).reshape(2, 2)
        >>> a
        [[0 1]
         [2 3]]
        >>> rm.amax(a, axis=1)
        [ 1.  3.]
        >>>
        >>> rm.amax(a, axis=0)
        [ 2.  3.]
        >>> rm.amax(a, axis=0, keepdims=True)
        [[ 2.  3.]]
        >>>
        >>> # Calculation of differentiation
        >>> va = rm.Variable(a)
        >>> out = rm.amax(va)
        >>> grad = out.grad()
        >>> grad.get(va) # Getting the gradient of 'va'.
        [[ 0.,  0.],
         [ 0.,  1.]]
    """
    pass


class mean(Node):
    '''
    This function calculates the mean of matrix elements.
    If the argument 'axis' is passed, this function performs
    mean calculation along the specified axis.

    Args:
        array (Node): Input array.
        axis (int): Calculate the mean along this axis
        keepdims (bool): If this is True, dimension will not be reduced.

    Returns:
        (Node): Mean array.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>>
        >>> x = np.random.rand(2, 3)
        >>> z = rm.mean(x)
        >>> z
    '''

    @classmethod
    def _oper_cpu(cls, arg, axis=None, keepdims=False):
        return np.mean(arg, axis=axis, keepdims=keepdims)

    @classmethod
    def _oper_gpu(cls, arg, axis=None, keepdims=False):
        if isinstance(axis, (int, tuple, type(None))):
            if isinstance(axis, tuple):
                size = 1
                for r in range(len(arg.shape)):
                    if r in axis:
                        size *= arg.shape[r]
            else:
                size = np.size(arg, axis)
            if not keepdims:
                if axis is None:
                    newshape = ()
                elif isinstance(axis, tuple):
                    temp_l = []
                    for r in range(len(arg.shape)):
                        if r not in axis:
                            temp_l.append(arg.shape[r])
                    newshape = tuple(temp_l)
                else:
                    newshape = arg.shape[:axis] + arg.shape[axis + 1:]
            else:
                axis_list = list(arg.shape)
                if axis is None:
                    newshape = tuple([1 for e in list(axis_list)])
                elif isinstance(axis, tuple):
                    for e in axis:
                        axis_list[e] = 1
                    newshape = tuple(axis_list)
                else:
                    axis_list[axis] = 1
                    newshape = tuple(axis_list)
            ret = GPUValue(shape=newshape)
            cudiv(cusum(get_gpu(arg), axis=axis, keepdims=keepdims), size, ret)
        return ret

    def __new__(cls, arg, axis=None, keepdims=False):
        value = cls.calc_value(arg, axis, keepdims=keepdims)
        ret = super(mean, cls).__new__(cls, value)
        ret.attrs._axis = axis
        ret.attrs._arg = arg
        ret.attrs._keep = keepdims
        return ret

    def new_expand_dims(self, a, axis):
        # if int is passed, retain the same behaviour
        if type(axis) == int:
            return np.expand_dims(a, axis)
        # insert axes to given indices
        for ax in sorted(axis):
            a = np.expand_dims(a, ax)
        return a

    def calc_size(self, a, axis):
        ax_list = list(a.shape)
        if isinstance(axis, int):
            return ax_list[axis]
        elif isinstance(axis, tuple):
            size = 0
            for i in axis:
                size += ax_list[i]
            return size

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            arg = self.attrs._arg
            axis = self.attrs._axis
            if axis is None:
                dx = np.ones_like(arg) * dy / np.size(arg)
            else:
                if not self.attrs._keep:
                    if isinstance(axis, (tuple, int)):
                        expanded = self.new_expand_dims(dy, axis)
                        size = self.calc_size(arg, axis)
                        dx = np.ones_like(arg) * expanded / size
                else:
                    if isinstance(axis, (tuple, int)):
                        size = self.calc_size(arg, axis)
                        dx = np.ones_like(arg) * dy / size
            arg._update_diff(context, dx, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            arg = self.attrs._arg
            axis = self.attrs._axis
            if axis is None:
                dx = get_gpu(arg).ones_like_me() * get_gpu(dy) / get_gpu(arg).size
            else:
                dy = get_gpu(dy).new_array()
                if not self.attrs._keep:
                    if isinstance(axis, (tuple, int)):
                        expanded = self.new_expand_dims(dy, axis)
                        dx = np.ones_like(arg, dtype=arg.dtype) * \
                            expanded / get_gpu(self.calc_size(arg, axis))
                else:
                    if isinstance(axis, (tuple, int)):
                        dx = np.ones_like(arg, dtype=arg.dtype) * dy / get_gpu(self.calc_size(arg, axis))
            arg._update_diff(context, get_gpu(dx), **kwargs)
