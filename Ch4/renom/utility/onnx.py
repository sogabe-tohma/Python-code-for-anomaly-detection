from __future__ import print_function, division, absolute_import

import collections
import onnx.helper
import onnx.numpy_helper
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

import renom
from ._onnx_impl import MODELDEFS, NODEDEFS, _ModelNode, _to_param_name, OBJNAMES


def _register_node(onnx_nodes, inputs, outputs, values, node):
    if isinstance(node, _ModelNode):
        node.register_params(onnx_nodes, inputs, outputs, values)
        return
    f = NODEDEFS.get(type(node))
    if f:
        f(onnx_nodes, inputs, outputs, values, node)
    elif isinstance(node, renom.Node):
        values.add(node.__class__.__name__, node)
    else:
        raise ValueError('Unknown node: %s' % type(node))


class _IdDict(dict):
    def add(self, obj):
        self[id(obj)] = obj


class _NamedIdDict(dict):
    def add(self, name, obj):
        k = id(obj)
        if k not in OBJNAMES:
            OBJNAMES[k] = "%s_%s" % (name, id(obj))
        self[id(obj)] = obj


class _OnnxHook:
    def call_enter(self, model, x, args, kwargs):
        return x, args, kwargs

    def call_leave(self, model, ret, x, args, kwargs):
        return ret

    def on_forward(self, model, forward, x, args, kwargs):
        output = forward(x, *args, **kwargs)
        conv = MODELDEFS.get(type(model), None)
        if conv:
            return conv(model, output, x)

        return output

    def leave_create(self, nodecls, ret):
        return ret


def _value_info(value):
    return onnx.helper.make_tensor_value_info(
        _to_param_name(value),
        NP_TYPE_TO_TENSOR_TYPE[value.dtype],
        value.shape)


def export_onnx(name, model, x, path, printtext=False):
    """
    This function exports an onnx file

    Args:
        name(str): The name of computational graph.
        model(Model): Neural Network Model
        x(ndarray): Dummy input for building a computational graph.
        path(str): The onnx file path to which the model will be export.
        printtext(bool): If True is given, this function print the str(model).

    """
    OBJNAMES.clear()

    if not isinstance(x, renom.Variable):
        x = renom.Variable(x)

    hook = _OnnxHook()
    renom.Model.set_hook(hook)
    renom.Node.set_hook(hook)

    try:
        with model.train():
            ret = model(x)
    finally:
        renom.Model.set_hook(None)

    cur = [ret]
    parent_nodes = collections.defaultdict(set)
    child_nodes = collections.defaultdict(set)

    nodes = _IdDict()
    roots = _IdDict()

    # build tree
    while cur:
        node = cur.pop(0)
        if not isinstance(node, renom.Node):
            continue

        parents = [n for n in node._get_graph() if isinstance(n, renom.Node)]
        if not parents:
            roots.add(node)

        cur.extend(parents)
        nodes.add(node)

        for parent in parents:
            nodes.add(parent)
            parent_nodes[id(node)].add(id(parent))
            child_nodes[id(parent)].add(id(node))

    # sort tree
    sorted = []
    remains = list(roots.values())
    while remains:
        node = remains.pop(0)
        sorted.append(node)

        children = child_nodes[id(node)]
        for child in children:
            parents = parent_nodes[child]
            parents.remove(id(node))
            if not parents:
                remains.append(nodes[child])

    # sort extract params

    OBJNAMES[id(x)] = 'input'
    inputs = _NamedIdDict()
    inputs.add('input', x)

    OBJNAMES[id(ret)] = 'output'
    outputs = _NamedIdDict()
    outputs.add('output', ret)

    onnx_nodes = []

    values = _NamedIdDict()
    for node in sorted:
        if node is not x:
            _register_node(onnx_nodes, inputs, outputs, values, node)

    if id(x) in values:
        del values[id(x)]

    inputs = [_value_info(v) for v in inputs.values()]
    outputs = [_value_info(v) for v in outputs.values()]

    for v in values.values():
        if isinstance(v, renom.Node):
            v.to_cpu()

    initializers = [onnx.numpy_helper.from_array(v, _to_param_name(v))
                    for v in values.values()]

    onnx_graph = onnx.helper.make_graph(
        onnx_nodes, name, inputs, outputs, initializer=initializers)

    model = onnx.helper.make_model(
        onnx_graph,
        producer_name='renom',
        producer_version=renom.__version__
    )

    with open(path, 'wb') as f:
        f.write(model.SerializeToString())

    if printtext:
        print(model)
