import inspect
from collections import namedtuple, defaultdict
from functools import partial
import functools
from itertools import chain, count, islice as take

#####################
# graph building
#####################

import IPython.display
def identity(value): return value


def build_graph(net, path_map='_'.join):
    net = {path: node if len(node) is 3 else (*node, None)
           for path, node in path_iter(net)}
    default_inputs = chain([('input',)], net.keys())
    def resolve_path(path, pfx): return pfx+path if (pfx +
                                                     path in net or not pfx) else resolve_path(net, path, pfx[:-1])
    return {path_map(path): (typ, value, ([path_map(default)] if inputs is None else [path_map(resolve_path(make_tuple(k), path[:-1])) for k in inputs]))
            for (path, (typ, value, inputs)), default in zip(net.items(), default_inputs)}


#####################
# network visualisation (requires pydot)
#####################


class ColorMap(dict):
    palette = (
        'bebada,ffffb3,fb8072,8dd3c7,80b1d3,fdb462,b3de69,fccde5,bc80bd,ccebc5,ffed6f,1f78b4,33a02c,e31a1c,ff7f00,'
        '4dddf8,e66493,b07b87,4e90e3,dea05e,d0c281,f0e189,e9e8b1,e0eb71,bbd2a4,6ed641,57eb9c,3ca4d4,92d5e7,b15928'
    ).split(',')

    def __missing__(self, key):
        self[key] = self.palette[len(self) % len(self.palette)]
        return self[key]


def make_pydot(nodes, edges, direction='LR', sep='_', **kwargs):
    from pydot import Dot, Cluster, Node, Edge

    class Subgraphs(dict):
        def __missing__(self, path):
            *parent, label = path
            subgraph = Cluster(sep.join(path), label=label,
                               style='rounded, filled', fillcolor='#77777744')
            self[tuple(parent)].add_subgraph(subgraph)
            return subgraph
    g = Dot(rankdir=direction, directed=True, **kwargs)
    g.set_node_defaults(
        shape='box', style='rounded, filled', fillcolor='#ffffff')
    subgraphs = Subgraphs({(): g})
    for path, attr in nodes:
        *parent, label = path.split(sep)
        subgraphs[tuple(parent)].add_node(
            Node(name=path, label=label, **attr))
    for src, dst, attr in edges:
        g.add_edge(Edge(src, dst, **attr))
    return g


class DotGraph():
    colors = ColorMap()

    def __init__(self, graph, size=15, direction='LR'):
        self.nodes = [(k, {
            'tooltip': '%s %.1000r' % (typ, value),
            'fillcolor': '#'+self.colors[typ],
        }) for k, (typ, value, inputs) in graph.items()]
        self.edges = [(src, k, {}) for (k, (_, _, inputs))
                      in graph.items() for src in inputs]
        self.size, self.direction = size, direction

    def dot_graph(self, **kwargs):
        return make_pydot(self.nodes, self.edges, size=self.size,
                          direction=self.direction, **kwargs)

    def svg(self, **kwargs):
        return self.dot_graph(**kwargs).create(format='svg').decode('utf-8')

    try:
        import pydot

        def _repr_svg_(self):
            return self.svg()
    except ImportError:
        def __repr__(self):
            return 'pydot is needed for network visualisation'
