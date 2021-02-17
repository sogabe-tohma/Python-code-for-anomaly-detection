from __future__ import division
from renom.core import Node
import weakref
import numpy as np
import renom.cuda
if renom.cuda.has_cuda():
    from renom.cuda.thrust.thrust import *
    from renom.cuda.gpuvalue.gpuvalue import GPUValue, get_gpu


def to_value(array):
    if isinstance(array, Node):
        array.to_cpu()
        return array.view(np.ndarray)
    elif renom.cuda.has_cuda() and isinstance(array, GPUValue):
        return array.new_array()
    else:
        return array


class UnaryOp(Node):
    def __new__(cls, arg, *args, **kwargs):
        value = cls.calc_value(arg, *args, **kwargs)
        ret = super(UnaryOp, cls).__new__(cls, value)
        ret.attrs._arg = arg
        return ret


class Neg(UnaryOp):

    @classmethod
    def _oper_cpu(cls, arg):
        return np.ndarray.__neg__(arg)

    @classmethod
    def _oper_gpu(cls, arg):
        return -(get_gpu(arg))

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            self.attrs._arg._update_diff(context, -dy, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            self.attrs._arg._update_diff(context, -get_gpu(dy), **kwargs)


Node.__neg__ = lambda self: Neg(self)


class Abs(UnaryOp):

    @classmethod
    def _oper_cpu(cls, arg):
        return np.ndarray.__abs__(arg)

    @classmethod
    def _oper_gpu(cls, arg):
        new_ptr = get_gpu(arg).empty_like_me()
        cuabs_forward(get_gpu(arg), new_ptr)
        return new_ptr

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            arg = to_value(self.attrs._arg)
            mask = np.where(arg > 0, 1, -1)
            self.attrs._arg._update_diff(context, mask * dy, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            new_ptr = get_gpu(dy).empty_like_me()
            cuabs_backward(get_gpu(self.attrs._arg), new_ptr)
            self.attrs._arg._update_diff(context, new_ptr, **kwargs)


Node.__abs__ = lambda self: Abs(self)


class Invert(UnaryOp):

    @classmethod
    def _oper_cpu(cls, arg):
        return np.ndarray.__invert__(arg)

    def _backward_cpu(self, context, dy, **kwargs):
        self.attrs._arg._update_diff(context, dy, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        return self.attrs._backward_cpu(context, dy, **kwargs)


class BinOp(Node):
    GRAPH = ['_lhs', '_rhs']

    def __new__(cls, lhs, rhs, *args, **kwargs):
        value = cls.calc_value(lhs, rhs, *args, **kwargs)
        ret = super(BinOp, cls).__new__(cls, value)
        ret.attrs._lhs = lhs
        ret.attrs._rhs = rhs
        return ret


def broad_cast(hs, dy):
    if isinstance(hs, np.ndarray):
        shape = list(hs.shape)
        if hs.shape != dy.shape:
            axis = []
            while len(shape) != len(dy.shape):
                if len(shape) < len(dy.shape):
                    shape.insert(0, 1)
            for i, s in enumerate(shape):
                if s == 1:
                    axis.append(i)
            if axis:
                dy = np.sum(dy, axis=tuple(axis))
        dy = dy.reshape(hs.shape)
    return dy


def cu_broad_cast(hs, dy):
    if isinstance(hs, GPUValue):
        shape = list(hs.shape)
        if hs.shape != dy.shape:
            axis = []
            while len(shape) != len(dy.shape):
                if len(shape) < len(dy.shape):
                    shape.insert(0, 1)
            for i, s in enumerate(shape):
                if s == 1:
                    axis.append(i)
            if axis:
                dy = cusum(dy, axis=tuple(axis))
            dy = dy.reshape(hs.shape)
    return dy


class Add(BinOp):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__add__(lhs, rhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return get_gpu(lhs) + get_gpu(rhs)

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._rhs, Node):
            r_dx = broad_cast(self.attrs._rhs, dy)
            self.attrs._rhs._update_diff(context, r_dx, **kwargs)

        if isinstance(self.attrs._lhs, Node):
            l_dx = broad_cast(self.attrs._lhs, dy)
            self.attrs._lhs._update_diff(context, l_dx, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._rhs, Node):
            rhs = get_gpu(self.attrs._rhs)
            r_dx = cu_broad_cast(rhs, get_gpu(dy))
            self.attrs._rhs._update_diff(context, r_dx, **kwargs)

        if isinstance(self.attrs._lhs, Node):
            lhs = get_gpu(self.attrs._lhs)
            l_dx = cu_broad_cast(lhs, get_gpu(dy))
            self.attrs._lhs._update_diff(context, l_dx, **kwargs)


Node.__add__ = lambda self, other: Add(self, other)
Node.__iadd__ = lambda self, other: Add(self, other)


class RAdd(Add):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__radd__(rhs, lhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return get_gpu(rhs) + get_gpu(lhs)


Node.__radd__ = lambda self, other: RAdd(other, self)


class Sub(BinOp):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__sub__(lhs, rhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return get_gpu(lhs) - get_gpu(rhs)

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._lhs, Node):
            l_dx = broad_cast(self.attrs._lhs, dy)
            self.attrs._lhs._update_diff(context, l_dx, **kwargs)

        if isinstance(self.attrs._rhs, Node):
            r_dx = broad_cast(self.attrs._rhs, -dy)
            self.attrs._rhs._update_diff(context, r_dx, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._lhs, Node):
            lhs = get_gpu(self.attrs._lhs)
            rhs = get_gpu(self.attrs._rhs)

            new_l_dx = cu_broad_cast(lhs, get_gpu(dy))
            self.attrs._lhs._update_diff(context, new_l_dx, **kwargs)

        if isinstance(self.attrs._rhs, Node):
            lhs = get_gpu(self.attrs._lhs)
            rhs = get_gpu(self.attrs._rhs)

            new_r_dx = cu_broad_cast(rhs, -1 * get_gpu(dy))

            self.attrs._rhs._update_diff(context, new_r_dx, **kwargs)


Node.__sub__ = lambda self, other: Sub(self, other)
Node.__isub__ = lambda self, other: Sub(self, other)


class RSub(Sub):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__rsub__(rhs, lhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return - get_gpu(rhs) + get_gpu(lhs)


Node.__rsub__ = lambda self, other: RSub(other, self)


class Mul(BinOp):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__mul__(lhs, rhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return get_gpu(lhs) * get_gpu(rhs)

    def _backward_cpu(self, context, dy, **kwargs):

        lhs = to_value(self.attrs._lhs)
        rhs = to_value(self.attrs._rhs)

        if isinstance(self.attrs._rhs, Node):
            r_dx = broad_cast(rhs, lhs * dy)
            self.attrs._rhs._update_diff(context, r_dx, **kwargs)

        if isinstance(self.attrs._lhs, Node):
            l_dx = broad_cast(lhs, rhs * dy)
            self.attrs._lhs._update_diff(context, l_dx, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._rhs, Node):
            lhs = get_gpu(self.attrs._lhs)
            rhs = get_gpu(self.attrs._rhs)
            dxr = cu_broad_cast(rhs, lhs * get_gpu(dy))

            self.attrs._rhs._update_diff(context, dxr, **kwargs)

        if isinstance(self.attrs._lhs, Node):
            lhs = get_gpu(self.attrs._lhs)
            rhs = get_gpu(self.attrs._rhs)
            dxl = cu_broad_cast(lhs, rhs * get_gpu(dy))

            self.attrs._lhs._update_diff(context, dxl, **kwargs)


Node.__mul__ = lambda self, other: Mul(self, other)
Node.__imul__ = lambda self, other: Mul(self, other)


class RMul(Mul):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__rmul__(rhs, lhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return get_gpu(rhs) * get_gpu(lhs)


Node.__rmul__ = lambda self, other: RMul(other, self)


class Div(BinOp):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__div__(lhs, rhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return get_gpu(lhs) / get_gpu(rhs)

    def _backward_cpu(self, context, dy, **kwargs):

        lhs = to_value(self.attrs._lhs)
        rhs = to_value(self.attrs._rhs)

        if isinstance(self.attrs._lhs, Node):
            l_dx = broad_cast(lhs, dy / rhs)
            self.attrs._lhs._update_diff(context, l_dx, **kwargs)

        if isinstance(self.attrs._rhs, Node):
            n = (-1) * (rhs ** (-2))
            r_dx = broad_cast(rhs, lhs * n * dy)
            self.attrs._rhs._update_diff(context, r_dx, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):

        if isinstance(self.attrs._lhs, Node):
            lhs = get_gpu(self.attrs._lhs)
            rhs = get_gpu(self.attrs._rhs)

            dxl = cu_broad_cast(lhs, get_gpu(dy) / rhs)
            self.attrs._lhs._update_diff(context, dxl, **kwargs)

        if isinstance(self.attrs._rhs, Node):
            lhs = get_gpu(self.attrs._lhs)
            rhs = get_gpu(self.attrs._rhs)

            v = rhs ** (-2.0) * -1.0 * lhs * get_gpu(dy)
            dxr = cu_broad_cast(rhs, v)
            self.attrs._rhs._update_diff(context, dxr, **kwargs)


Node.__div__ = lambda self, other: Div(self, other)


class RDiv(Div):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__rdiv__(rhs, lhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return get_gpu(lhs) / get_gpu(rhs)


class TrueDiv(Div):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        ret = np.ndarray.__truediv__(lhs, rhs)
        return ret

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return get_gpu(lhs) / get_gpu(rhs)


Node.__truediv__ = lambda self, other: TrueDiv(self, other)
Node.__itruediv__ = lambda self, other: TrueDiv(self, other)


class RTrueDiv(TrueDiv):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__rtruediv__(rhs, lhs)


Node.__rtruediv__ = lambda self, other: RTrueDiv(other, self)


class Mod(BinOp):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__mod__(lhs, rhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return cls._oper_cpu(lhs, rhs)

    def _backward_cpu(self, context, dy, **kwargs):
        pass

    def _backward_gpu(self, context, dy, **kwargs):
        self._backward_cpu(context, dy, **kwargs)


class RMod(Mod):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__rmod__(rhs, lhs)


class DivMod(BinOp):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        d, m = np.ndarray.__divmod__(lhs, rhs)
        return np.array([d, m])

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return cls._oper_cpu(lhs, rhs)

    def _backward_cpu(self, context, dy, **kwargs):
        pass

    def _backward_gpu(self, context, dy, **kwargs):
        self._backward_cpu(context, dy, **kwargs)


class RDivMod(DivMod):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        d, m = np.ndarray.__rdivmod__(rhs, lhs)
        return np.array([d, m])


class Pow(BinOp):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__pow__(lhs, rhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return get_gpu(lhs) ** get_gpu(rhs)

    def _backward_cpu(self, context, dy, **kwargs):
        lhs = to_value(self.attrs._lhs)
        rhs = to_value(self.attrs._rhs)

        if isinstance(self.attrs._lhs, Node):
            self.attrs._lhs._update_diff(context, dy * (np.power(lhs, rhs - 1) * rhs), **kwargs)

        if isinstance(self.attrs._rhs, Node):
            self.attrs._rhs._update_diff(context, dy * self * np.log(lhs), **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):

        if isinstance(self.attrs._lhs, Node):
            lhs = get_gpu(self.attrs._lhs)
            rhs = get_gpu(self.attrs._rhs)

            v = get_gpu(dy) * rhs * (GPUValue.__pow__(lhs, rhs - 1))

            dxl = cu_broad_cast(lhs, v)
            self.attrs._lhs._update_diff(context, dxl, **kwargs)

        if isinstance(self.attrs._rhs, Node):
            lhs = get_gpu(self.attrs._lhs).empty_like_me()
            culoge(get_gpu(self.attrs._lhs), lhs)
            new_r_dx = get_gpu(dy) * get_gpu(self) * lhs
            self.attrs._rhs._update_diff(context, new_r_dx, **kwargs)


Node.__pow__ = lambda self, other: Pow(self, other)
Node.__ipow__ = lambda self, other: Pow(self, other)


class RPow(Pow):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__rpow__(rhs, lhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return get_gpu(lhs) ** get_gpu(rhs)


Node.__rpow__ = lambda self, other: RPow(other, self)


class Lshift(BinOp):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__lshift__(lhs, rhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return cls._oper_cpu(lhs, rhs)

    def _backward_cpu(self, context, dy, **kwargs):
        pass

    def _backward_gpu(self, context, dy, **kwargs):
        self._backward_cpu(context, dy, **kwargs)


class RLshift(Lshift):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__rlshift__(rhs, lhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return cls._oper_cpu(lhs, rhs)


class Rshift(BinOp):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__rshift__(lhs, rhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return cls._oper_cpu(lhs, rhs)

    def _backward_cpu(self, context, dy, **kwargs):
        pass

    def _backward_gpu(self, context, dy, **kwargs):
        self._backward_cpu(context, dy, **kwargs)


class RRshift(Lshift):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__rrshift__(rhs, lhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return cls._oper_cpu(lhs, rhs)


class And(BinOp):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__and__(lhs, rhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return cls._oper_cpu(lhs, rhs)

    def _backward_cpu(self, context, dy, **kwargs):
        pass

    def _backward_gpu(self, context, dy, **kwargs):
        self._backward_cpu(context, dy, **kwargs)


class RAnd(Lshift):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__rand__(rhs, lhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return cls._oper_cpu(lhs, rhs)

    def _backward_gpu(self, context, dy, **kwargs):
        self._backward_cpu(context, dy, **kwargs)


class Xor(BinOp):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__xor__(lhs, rhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return cls._oper_cpu(lhs, rhs)

    def _backward_cpu(self, context, dy, **kwargs):
        pass

    def _backward_gpu(self, context, dy, **kwargs):
        self._backward_cpu(context, dy, **kwargs)


class RXor(Lshift):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__rxor__(rhs, lhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return cls._oper_cpu(lhs, rhs)


class Or(BinOp):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__or__(lhs, rhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return cls._oper_cpu(lhs, rhs)

    def _backward_cpu(self, context, dy, **kwargs):
        pass

    def _backward_gpu(self, context, dy, **kwargs):
        self._backward_cpu(context, dy, **kwargs)


class ROr(Lshift):

    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__ror__(rhs, lhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return cls._oper_cpu(lhs, rhs)


class GetItem(BinOp):
    @classmethod
    def _oper_cpu(cls, lhs, rhs):
        return np.ndarray.__getitem__(lhs, rhs)

    @classmethod
    def _oper_gpu(cls, lhs, rhs):
        return get_gpu(lhs)[rhs]

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._lhs, Node):
            zero = np.zeros_like(to_value(self.attrs._lhs))
            np.add.at(zero, self.attrs._rhs, to_value(dy))
            self.attrs._lhs._update_diff(context, zero, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._lhs, Node):
            if self._is_advanced_indexing(self.attrs._lhs, self.attrs._rhs):
                self._backward_cpu(context, to_value(dy), **kwargs)
            else:
                zero = get_gpu(self.attrs._lhs).zeros_like_me()
                zero[self.attrs._rhs] = dy
                self.attrs._lhs._update_diff(context, zero, **kwargs)

    def _is_advanced_indexing(self, array, index):
        if isinstance(index, (int, slice, type(None), type(Ellipsis))):
            return False
        elif isinstance(index, tuple):
            if all([isinstance(o, (int, slice, type(None), type(Ellipsis))) for o in index]):
                return False
        elif isinstance(index, np.ndarray):
            if index.dtype == np.bool:
                return False
        return True


Node.__getitem__ = lambda self, index: GetItem(self, index)


class GetFgAry(Node):
    @classmethod
    def _oper_cpu(cls, arg):
        return arg[:, :, 1, :, :]

    @classmethod
    def _oper_gpu(cls, arg):
        shape = arg.shape
        fg_ary = GPUValue(shape=(shape[0], shape[1], 1, shape[3], shape[4]))
        arg = get_gpu(arg)
        cu_get_fg_ary_forward(arg, fg_ary)
        return fg_ary

    def __new__(cls, arg):
        value = cls.calc_value(arg)
        ret = super(GetFgAry, cls).__new__(cls, value)
        ret.attrs._arg = arg
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            zero = np.zeros_like(np.array(self.attrs._arg))
            zero[:, :, 1, :, :] = np.array(dy)
            self.attrs._arg._update_diff(context, zero, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._lhs, Node):
            zero = get_gpu(self.attrs._lhs).zeros_like_me()
            cu_get_fg_ary_backward(dy, zero)
            self.attrs._arg._update_diff(context, zero, **kwargs)


class GetIthAry(Node):
    @classmethod
    def _oper_cpu(cls, arg, i):
        return arg[i]

    @classmethod
    def _oper_gpu(cls, arg, i):
        shape = arg.shape
        ith_ary = GPUValue(shape=(shape[1:]))
        arg = get_gpu(arg)
        cu_get_ith_ary_forward(arg, ith_ary, i)
        return ith_ary

    def __new__(cls, arg, i):
        value = cls.calc_value(arg, i)
        ret = super(GetIthAry, cls).__new__(cls, value)
        ret.attrs._arg = arg
        ret._index = i
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            zero = np.zeros_like(np.array(self.attrs._arg))
            zero[self.attrs._index] = np.array(dy)
            self.attrs._arg._update_diff(context, zero, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._lhs, Node):
            zero = get_gpu(self.attrs._lhs).zeros_like_me()
            cu_get_ith_ary_backward(dy, zero, self.attrs._index)
            self.attrs._arg._update_diff(context, zero, **kwargs)


class GetNthAry(Node):
    def __new__(cls, arg, i, j):
        value = cls.calc_value(arg, i, j)
        ret = super(GetNthAry, cls).__new__(cls, value)
        return ret

    @classmethod
    def _oper_gpu(cls, arg, i, j):
        ary = GPUValue(shape=(arg.shape[0], ((arg.shape[1] - (i + 1)) // j) + 1))
        arg = get_gpu(arg)
        cu_get_every_nth_ary(arg, ary, i, j)
        return ary


class GetSlice(Node):
    @classmethod
    def _oper_cpu(cls, arg, i, j):
        return np.ndarray.__getslice__(arg, i, j)

    @classmethod
    def _oper_gpu(cls, arg, i, j):
        return cls._oper_cpu(arg, i, j)

    def __new__(cls, arg, i, j):
        value = cls.calc_value(arg, i, j)
        ret = super(GetSlice, cls).__new__(cls, value)
        ret.attrs._arg = arg
        ret.attrs._i, ret.attrs._j = i, j
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            zero = np.zeros_like(np.array(self.attrs._arg))
            zero[self.attrs._i:self.attrs._j] = np.array(dy)
            self.attrs._arg._update_diff(context, zero, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        self._backward_cpu(context, dy, **kwargs)


Node.__getslice__ = lambda self, i, j: GetSlice(self, i, j)


class AssignPredBox(Node):
    def __new__(cls, arg, x, y, h, w):
        ary = GPUValue(shape=arg.shape)
        x = get_gpu(x)
        y = get_gpu(y)
        h = get_gpu(h)
        w = get_gpu(w)
        value = cls.calc_value(ary, x, y, h, w)
        ret = super(AssignPredBox, cls).__new__(cls, value)
        return ret

    @classmethod
    def _oper_gpu(cls, ary, x, y, h, w):
        cu_assign_pred_box(x, y, h, w, ary)
        return ary


class PredCtr(Node):
    def __new__(cls, arg, length, ctr):
        ary = GPUValue(shape=arg.shape)
        arg = get_gpu(arg)
        length = get_gpu(length)
        ctr = get_gpu(ctr)
        value = cls.calc_value(arg, length, ctr, ary)
        ret = super(PredCtr, cls).__new__(cls, value)
        return ret

    @classmethod
    def _oper_gpu(cls, arg, length, ctr, ary):
        cu_pred_ctr(arg, length, ctr, ary)
        return ary


class GetIthBbox(Node):
    def __new__(cls, arg, i):
        arg = get_gpu(arg)
        ary = GPUValue(shape=(arg.shape[0], 1))
        value = cls.calc_value(arg, i, ary)
        ret = super(GetIthBbox, cls).__new__(cls, value)
        return ret

    @classmethod
    def _oper_gpu(cls, arg, i, ary):
        cu_get_ith_bbox(arg, i, ary)
        return ary


class Reshape(Node):

    @classmethod
    def _oper_cpu(cls, array, shape):
        return np.reshape(array, shape).copy()

    @classmethod
    def _oper_gpu(cls, array, shape):
        return get_gpu(array).reshape(shape)

    def __new__(cls, array, shape):
        value = cls.calc_value(array, shape)
        ret = super(Reshape, cls).__new__(cls, value)
        ret.attrs._array = array
        ret.attrs._shape = array.shape
        ret._shape_to = shape
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._array, Node):
            self.attrs._array._update_diff(context, to_value(
                dy).reshape(self.attrs._shape), **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._array, Node):
            self.attrs._array._update_diff(context, get_gpu(
                dy).reshape(self.attrs._shape), **kwargs)


def _reshape(self, *shape):
    """Returns reshaped array.

    Args:
        shape(list, int): Array will be reshaped according to given shape.

    Returns:
        (Node): Reshaped array.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> a = rm.Variable(np.arange(4).reshape(2, 2))
        >>> print(a)
        [[ 0.  1.]
         [ 2.  3.]]
        >>> print(a.reshape(-1))
        [ 0.  1.  2.  3.]
        >>> print(a.reshape(1, 4))
        [[ 0.  1.  2.  3.]]
    """
    if isinstance(shape[0], (list, tuple)):
        shape = tuple(shape[0])
    return Reshape(self, shape)


Node.reshape = _reshape


class Transpose2d(UnaryOp):
    @classmethod
    def _oper_cpu(cls, arg):
        assert(len(arg.shape) < 3)
        return to_value(arg).T

    @classmethod
    def _oper_gpu(cls, arg):
        return get_gpu(arg).T

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            self.attrs._arg._update_diff(context, to_value(dy).T, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            self.attrs._arg._update_diff(context, get_gpu(dy).T, **kwargs)


Node.T = property(lambda self: Transpose2d(self))


class Transpose(Node):

    @classmethod
    def _oper_cpu(cls, arg, axis):
        return np.transpose(to_value(arg), axis)

    @classmethod
    def _oper_gpu(cls, arg, axis):
        return get_gpu(arg).transpose(axis)

    def __new__(cls, arg, axis):
        value = cls.calc_value(arg, axis)
        ret = super(Transpose, cls).__new__(cls, value)
        rev = [-1] * len(axis)
        for i, a in enumerate(axis):
            rev[a] = i
        ret.attrs._arg = arg
        ret.attrs._axis = rev
        return ret

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            axis = self.attrs._axis
            self.attrs._arg._update_diff(context, to_value(dy).transpose(axis), **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            axis = self.attrs._axis
            self.attrs._arg._update_diff(context, get_gpu(dy).transpose(axis), **kwargs)


def _transpose(self, *axis):
    """Returns an array with axes transposed.

    Args:
        axes(list of ints): Permute the axes according to the values given.

    Returns:
        (Node): Transposed array.

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> a = rm.Variable(np.arange(4).reshape(2, 2))
        >>> print(a)
        [[ 0.  1.]
         [ 2.  3.]]
        >>> print(a.transpose(1, 0))
        [[ 0.  2.]
         [ 1.  3.]]

    """
    ax = axis
    if isinstance(ax[0], (tuple, list)):
        ax = ax[0]
    else:
        ax = tuple(axis)

    assert len(self.shape) == len(ax), "Axis must be same size to matrix dim size."
    return Transpose(self, ax)


Node.transpose = _transpose


class Pos(UnaryOp):

    @classmethod
    def _oper_cpu(cls, arg):
        return np.ndarray.__pos__(arg)

    @classmethod
    def _oper_gpu(cls, arg):
        return +get_gpu(arg)

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            self.attrs._arg._update_diff(context, dy, **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._arg, Node):
            self.attrs._arg._update_diff(context, get_gpu(dy), **kwargs)


class Mark(Pos):
    def __new__(cls, arg, model):
        ret = super(Mark, cls).__new__(cls, arg)
        ret.modelref = weakref.ref(model)
        return ret

    def _reduce_graph(self):
        return

    @classmethod
    def _run_node_hook(cls, ret):
        return ret


class NodeMark(Mark):
    pass


class ModelMark(Mark):
    pass


class EnterModel(ModelMark):
    pass


class LeaveModel(ModelMark):
    pass
