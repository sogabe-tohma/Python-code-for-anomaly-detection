from __future__ import absolute_import, division
import numpy as np
import onnx

import renom

MODELDEFS = {}
NODEDEFS = {}
OBJNAMES = {}


def _to_param_name(obj):
    name = OBJNAMES.get(id(obj), None)
    if not name:
        name = "%s_%s" % (obj.__class__.__name__, id(obj))
    return name


def modeldef(target):
    def wrapper(cls):
        MODELDEFS[target] = cls
        return cls

    return wrapper


class _ModelNode(renom.Node):
    def __new__(cls, model, output, input):
        ret = super(_ModelNode, cls).__new__(cls, output)
        ret.attrs.input = input
        ret.model = model
        ret.output = output

        return ret


@modeldef(renom.Dense)
class DenseModel(_ModelNode):
    def register_params(self, onnx_nodes, inputs, outputs, values):
        w = self.model.params["w"]
        inputs.add("w", w)
        values.add("w", w)

        b = self.model.params["b"]
        inputs.add("b", b)
        values.add("b", b)

        onnx_nodes.append(
            onnx.helper.make_node(
                'Gemm',
                [_to_param_name(v) for v in (self.attrs.input, w, b)],
                [_to_param_name(self)]))


@modeldef(renom.Conv2d)
class Conv2dModel(_ModelNode):
    def register_params(self, onnx_nodes, inputs, outputs, values):
        w = self.model.params["w"]
        inputs.add("w", w)
        values.add("w", w)

        b = self.model.params["b"]
        b = b.reshape(np.prod(b.shape))

        inputs.add("b", b)
        values.add("b", b)

        onnx_nodes.append(
            onnx.helper.make_node(
                'Conv',
                [_to_param_name(v) for v in (self.attrs.input, w, b)],
                [_to_param_name(self)],
                kernel_shape=self.output.attrs._kernel,
                pads=self.output.attrs._padding,
                strides=self.output.attrs._stride,
                dilations=self.output.attrs._dilation))


def nodedef(target):
    def wrapper(f):
        NODEDEFS[target] = f
        return f
    return wrapper


def _register_unary(onnx_nodes, inputs, outputs, values, node, name):
    onnx_nodes.append(
        onnx.helper.make_node(
            name,
            [_to_param_name(node.attrs._arg)],
            [_to_param_name(node)]))


@nodedef(renom.core.basic_ops.Neg)
def register_neg(onnx_nodes, inputs, outputs, values, node):
    _register_unary(onnx_nodes, inputs, outputs, values, node, 'Neg')


@nodedef(renom.core.basic_ops.Abs)
def register_abs(onnx_nodes, inputs, outputs, values, node):
    _register_unary(onnx_nodes, inputs, outputs, values, node, 'Abs')


def _register_binop(onnx_nodes, inputs, outputs, values, node, name):
    onnx_nodes.append(
        onnx.helper.make_node(
            name,
            [_to_param_name(node.attrs._lhs), _to_param_name(node.attrs._rhs)],
            [_to_param_name(node)]))


@nodedef(renom.core.basic_ops.Add)
def register_add(onnx_nodes, inputs, outputs, values, node):
    _register_binop(onnx_nodes, inputs, outputs, values, node, 'Add')


@nodedef(renom.core.basic_ops.Sub)
def register_sub(onnx_nodes, inputs, outputs, values, node):
    _register_binop(onnx_nodes, inputs, outputs, values, node, 'Sub')


@nodedef(renom.core.basic_ops.Mul)
def register_mul(onnx_nodes, inputs, outputs, values, node):
    _register_binop(onnx_nodes, inputs, outputs, values, node, 'Mul')


@nodedef(renom.core.basic_ops.TrueDiv)
def register_div(onnx_nodes, inputs, outputs, values, node):
    _register_binop(onnx_nodes, inputs, outputs, values, node, 'Div')


@nodedef(renom.relu)
def register_relu(onnx_nodes, inputs, outputs, values, node):
    onnx_nodes.append(
        onnx.helper.make_node(
            'Relu',
            [_to_param_name(node.attrs._arg)],
            [_to_param_name(node)]))


@nodedef(renom.max_pool2d)
def register_max_pool2d(onnx_nodes, inputs, outputs, values, node):
    onnx_nodes.append(
        onnx.helper.make_node(
            'MaxPool',
            [_to_param_name(node.attrs._x)],
            [_to_param_name(node)],
            kernel_shape=node.attrs._kernel,
            pads=node.attrs._padding,
            strides=node.attrs._stride,
        ))


@nodedef(renom.dropout)
def register_dropout(onnx_nodes, inputs, outputs, values, node):
    onnx_nodes.append(
        onnx.helper.make_node(
            'Dropout',
            [_to_param_name(node.attrs._x)],
            [_to_param_name(node)],
            ratio=node._ratio))


@nodedef(renom.Reshape)
def register_reshape(onnx_nodes, inputs, outputs, values, node):
    shape = np.array(node._shape_to)
    inputs.add('shape', shape)
    values.add('shape', shape)

    onnx_nodes.append(
        onnx.helper.make_node(
            'Reshape',
            [_to_param_name(node.attrs._array), _to_param_name(shape)],
            [_to_param_name(node)]))
