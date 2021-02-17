from __future__ import absolute_import
import itertools
import collections
import weakref
import renom

try:
    from graphviz import Digraph, Graph
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


def GET_ACTIVE_NODE():
    global ACTIVE_NODE
    return ACTIVE_NODE


def SET_NODE_DICT(id, val):
    global ACTIVE_NODE
    ACTIVE_NODE[id] = val


def GET_ACTIVE_GPU():
    global ACTIVE_GPU
    return ACTIVE_GPU


def SET_GPU_DICT(id, val):
    global ACTIVE_GPU
    ACTIVE_GPU[id] = val


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
        if not hasattr(o, "attrs"):
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


class _Box:
    @staticmethod
    def create(obj):
        target = obj.modelref()
        if isinstance(target, renom.Model):
            return _ModelBox(obj)
        else:
            return _Box(obj)

    def __init__(self, obj):
        self.nexts = set()
        self.obj = obj
        self.join = None

    def addnext(self, nextbox):
        self.nexts.add(nextbox)

    def joinnext(self):
        pass

    def nodename(self):
        if self.join:
            return str(id(self.join.obj))
        else:
            return str(id(self.obj))

    def create_node(self, context, graph):
        obj = self.obj.modelref()

        shape = 'diamond'
        color = 'gray'
        label = obj.__class__.__name__

        graph.node(str(id(self.obj)), label=label, shape=shape,
                   style='filled', fillcolor=color, color='black')

    def create_edge(self, context):
        f = self.nodename()
        for c in self.nexts:
            if c.join is self:
                continue
            t = c.nodename()
            context.root.graph.edge(f, t)


class _ModelBox(_Box):
    def create_node(self, context, graph):
        if self.join:
            return

        model = self.obj.modelref()

        modelinfo = context.get_modelinfo(model)
        if modelinfo.children:
            shape = 'circle'
            color = 'gray'
            if isinstance(self.obj, renom.core.EnterModel):
                label = 'S'
            else:
                label = 'E'
        else:
            shape = 'box'
            color = 'white'
            name = context.get_modelinfo(model).name
            label = '%s(%s)' % (name, type(model).__name__)
            color = 'white'

        graph.node(str(id(self.obj)), label=label, shape=shape,
                   style='filled', fillcolor=color, color='black')

    def joinnext(self):
        model = self.obj.modelref()
        for c in self.nexts:
            if c.obj.modelref() is model:
                c.join = self


class _ModelInfo:
    def __init__(self, parent, name, model):
        self.parent = parent
        self.children = weakref.WeakSet()
        if parent:
            self.parent.children.add(self)

        self.nodes = []

        self.name = name
        self.model = model
        self.enter = self.leave = None
        self.graph = None

    def create_graph(self, context):
        self.graph = Digraph(name='cluster=' + self.name)
        self.graph.attr(label='%s(%s)' % (self.name, self.model.__class__.__name__),
                        labelloc='top', labeljust='left')

        for node in self.nodes:
            node.create_node(context, self.graph)

    def addnode(self, node):
        self.nodes.append(node)

    def setbox(self, box):
        if isinstance(box.obj, renom.core.EnterModel):
            self.enter = box
        elif isinstance(box.obj, renom.core.LeaveModel):
            self.leave = box
        else:
            raise ValueError()


class ModelGraphContext:

    def __init__(self):
        pass

    def get_modelinfo(self, model):
        return self.models.get(id(model))

    def walk_model(self, model):
        self.models = {}

        models = [(None, 'root', model)]

        while models:
            parent, name, model = models.pop()
            p = _ModelInfo(parent, name, model)
            if not parent:
                self.root = p

            self.models[id(model)] = p

            models.extend((p, k, v) for k, v in model.get_model_children())

    def selectmodel(self, node, curmodel):
        target = node.modelref()
        modelinfo = self.models.get(id(target))
        if not modelinfo:
            if curmodel is not None:
                modelinfo = self.models.get(id(curmodel.modelref()))

                if isinstance(curmodel, renom.core.LeaveModel):
                    modelinfo = modelinfo
                else:
                    modelinfo = modelinfo.parent
        else:
            if not modelinfo.children:
                if modelinfo.parent:
                    modelinfo = modelinfo.parent

        if not modelinfo:
            modelinfo = self.root

        return modelinfo

    def getbox(self, node, nextbox, curmodel):
        nodeid = id(node)
        target = node.modelref()
        modelinfo = self.models.get(id(target))
        if modelinfo:
            if isinstance(node, renom.core.EnterModel):
                if modelinfo.enter:
                    return modelinfo.enter
            if isinstance(node, renom.core.LeaveModel):
                if modelinfo.leave:
                    return modelinfo.leave

        parentmodel = self.selectmodel(node, curmodel)
        if parentmodel.children:
            if nodeid not in self.boxes:
                box = _Box.create(node)
                self.boxes[nodeid] = box

                if modelinfo:
                    modelinfo.setbox(box)

                parentmodel.addnode(box)

            return self.boxes[nodeid]

    def walk_node(self, node):
        self.boxes = {}
        self._walk_node(node, None, None, set())
        for box in self.boxes.values():
            box.joinnext()

    def _walk_node(self, node, nextbox, curmodel, seen):
        if not isinstance(node, renom.core.Node):
            return

        if isinstance(node, renom.core.Mark):
            box = self.getbox(node, nextbox, curmodel)
            if box:
                if nextbox is not None:
                    box.addnext(nextbox)
                nextbox = box

        id_node = id(node)
        if id_node in seen:
            return
        seen.add(id_node)

        if not node.attrs:
            return

        if isinstance(node, renom.core.ModelMark):
            curmodel = node

        for attr in node.attrs.get_attrs():
            self._walk_node(attr, nextbox, curmodel, seen)

    def build_subgraph(self):

        # build pathes from root to leaf
        leafs = []
        q = collections.deque([(self.root, [])])
        while q:
            model, path = q.pop()
            path = [model, ] + path
            if not model.children:
                leafs.append(path)
            else:
                model.create_graph(self)
                for c in model.children:
                    q.append((c, path))

        # create sub graphs from leaf to root
        leafs.sort(key=len, reverse=True)
        seen = set()
        for leaf in leafs:
            while len(leaf) >= 2:
                child = leaf.pop(0)
                parent = leaf[0]
                if (child, parent) in seen:
                    break

                parent.graph.subgraph(child.graph)
                seen.add((child, parent))

        for box in self.boxes.values():
            box.create_edge(self)

    def build(self, nnmodel, value):
        self.walk_model(nnmodel)
        self.walk_node(value)
        self.build_subgraph()
        return self.root.graph


class GraphHook:
    def call_enter(self, model, x, args, kwargs):
        return renom.core.EnterModel(x, model), args, kwargs

    def call_leave(self, model, ret, x, args, kwargs):
        return renom.core.LeaveModel(ret, model)

    def on_forward(self, model, forward, x, args, kwargs):
        return forward(x, *args, **kwargs)

    def leave_create(self, nodecls, ret):
        ret = renom.core.NodeMark(ret, ret)
        return ret


def SET_MODEL_GRAPH(use):
    '''Specify if information to build model graph are generated.

    Args:
        use (bool): True if informations to build model graph are generated.

    Example:
        >>> import renom as rm
        >>> import numpy as np
        >>> x = np.random.rand(1, 3)
        array([[ 0.11871966  0.48498547  0.7406374 ]])
        >>> z = rm.softmax(x)
        softmax([[ 0.23229694  0.33505085  0.43265226]])
        >>> np.sum(z, axis=1)
        array([ 1.])

    '''

    if use:
        hook = GraphHook()
        renom.Model.set_hook(hook)
        renom.Node.set_hook(hook)
    else:
        renom.Model.set_hook(None)
        renom.Node.set_hook(None)


def showmark(cls):
    cls.SHOWMARK = True
    return cls


def BUILD_MODEL_GRAPH(model, value):
    '''Build model graph. Returns graph object of Graphviz.

    Args:
        model (Model): Root model object.
        value: result value of the model

    Example:
        >>> SET_MODEL_GRAPH(True)
        >>> model = MNist()
        >>> value = model(np.random.rand(10, 10))
        >>> SET_MODEL_GRAPH(False)
        >>> graph = BUILD_MODEL_GRAPH(model, value)
        >>> graph.view()
    '''

    c = ModelGraphContext()
    return c.build(model, value)
