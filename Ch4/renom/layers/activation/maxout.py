#!/usr/bin/env python
# -*- coding: utf-8 -*-
import renom as rm
#from renom.operation import concat, amax
from renom.core import Node
from renom.debug_graph import showmark


@showmark
class maxout(Node):

    def __new__(self, x, slice_size=1, axis=1):
        if axis is None:
            axis = 1
        assert len(x.shape) > 1
        input_length = x.shape[axis]
        maxes = []
        # TODO: Ensure that input_length is evenly divisible by _slice_size
        for u in range(input_length // slice_size):
            offset = u * slice_size
            maxes.append(rm.amax(x[:, offset:offset + slice_size],
                                 axis=axis, keepdims=True))
        return rm.concat(*maxes, axis=axis)


class Maxout:
    '''
    Maxout Network Actionvation Function as described at:
        http://proceedings.mlr.press/v28/goodfellow13.pdf

    :math:
        y = Output
        k = slice_size
        x = input
        y_i = max(x_((i*k)+ j), z=1..k)

    Args:
        x (ndarray, Node): Input numpy array or Node instance
        axis (Integer): Axis over which the input array should be sliced
        slice_size (Integer): Amount of units to compare for each output

        The output size will come out as x.shape[axis] // slice_size

    Example:
    >>> import numpy as np
    >>> import renom as rm
    >>> x = np.array([[1, -1]])
    >>> x
    array([[ 1, -1]])
    >>> rm.maxout(x)
    concat([[ 1., -1.]])
    >>> rm.maxout(x, slice_size=2)
    concat([[1.]])
    >>> activation = rm.Maxout(slice_size=2)
    >>> activation(x)
    concat([[1.]])
    >>>
    '''

    def __init__(self, slice_size=1, axis=1):
        self._axis = axis
        self._slice_size = slice_size

    def __call__(self, x):
        return maxout(x, self._slice_size, self._axis)
