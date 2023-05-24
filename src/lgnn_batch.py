import warnings
from typing import Any, Iterable, Mapping, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp


ArrayTree = Union[jnp.ndarray,
                  Iterable['ArrayTree'], Mapping[Any, 'ArrayTree']]


class GraphsTuple(NamedTuple):
    """Class for holding graph information.
    """
    nodes: Optional[ArrayTree]
    edges: Optional[ArrayTree]
    receivers: Optional[jnp.ndarray]  # with integer dtype
    senders: Optional[jnp.ndarray]  # with integer dtype
    globals: Optional[ArrayTree]
    n_node: jnp.ndarray  # with integer dtype
    n_edge: jnp.ndarray   # with integer dtype
    e_order: Optional[jnp.ndarray]
    e_mask: jnp.ndarray
    n_mask: jnp.ndarray


def pad_graph_with_edges(graph, max_edges):
    """Pad graph to accomodate max edges (as edges can be dynamic in MD simulations).

    :param graph: graph tuple
    :type graph: GraphsTuple
    :param max_edges: max number of edges 
    :type max_edges: int
    :return: padded graph tuple
    :rtype: GraphsTuple
    """
    try:
        return pad_with_graphs(graph, graph.n_node.sum()+1, max_edges+1)
    except Exception as excp:
        warnings.warn(str(excp))
        max_edges += 10
        return pad_graph_with_edges(graph, max_edges)


def mkgraph(*args, mass=None, max_edges, L=None, atoms=None, **kwargs):
    """Make graph with padding (edges).
    """
    # nodes = kwargs["nodes"]
    # nodes["mass"] = mass[nodes["type"]]
    graph = GraphsTuple(*args,
                        e_mask=jnp.ones(kwargs["senders"].shape, dtype=bool),
                        n_mask=jnp.ones(jnp.sum(kwargs["n_node"]), dtype=bool),
                        **kwargs)
    return pad_graph_with_edges(graph, max_edges)


def samegraph(*args, L=None, atoms=None, **kwargs):
    """make graph from dict.
    """
    graph = GraphsTuple(*args,
                        e_mask=jnp.ones(kwargs["senders"].shape, dtype=bool),
                        n_mask=jnp.ones(jnp.sum(kwargs["n_node"]), dtype=bool),
                        **kwargs)
    return graph


def _batch(graphs, np_):
    """Returns batched graph given a list of graphs and a numpy-like module."""
    # Calculates offsets for sender and receiver arrays, caused by concatenating
    # the nodes arrays.
    offsets = np_.cumsum(
        np_.array([0] + [np_.sum(g.n_node) for g in graphs[:-1]]))

    edge_order_offsets = np_.cumsum(
        np_.array([0] + [len(g.senders) for g in graphs[:-1]]))

    def _map_concat(nests):
        concat = lambda *args: np_.concatenate(args)
        # return jax.tree_multimap(concat, *nests)
        return jax.tree_map(concat, *nests)

    return GraphsTuple(
        n_node=np_.concatenate([g.n_node for g in graphs]),
        n_edge=np_.concatenate([g.n_edge for g in graphs]),
        nodes=_map_concat([g.nodes for g in graphs]),
        edges=_map_concat([g.edges for g in graphs]),
        e_mask=_map_concat([g.e_mask for g in graphs]),
        n_mask=_map_concat([g.n_mask for g in graphs]),
        e_order=_map_concat(
            [g.e_order + o for g, o in zip(graphs, edge_order_offsets)]),
        globals=_map_concat([g.globals for g in graphs]),
        senders=np_.concatenate(
            [g.senders + o for g, o in zip(graphs, offsets)]),
        receivers=np_.concatenate(
            [g.receivers + o for g, o in zip(graphs, offsets)]))


def pad_with_graphs(graph: GraphsTuple,
                    n_node: int,
                    n_edge: int,
                    n_graph: int = 2) -> GraphsTuple:
    """Pads a ``GraphsTuple`` to size by adding computation preserving graphs.
    The ``GraphsTuple`` is padded by first adding a dummy graph which contains the
    padding nodes and edges, and then empty graphs without nodes or edges.
    The empty graphs and the dummy graph do not interfer with the graphnet
    calculations on the original graph, and so are computation preserving.
    The padding graph requires at least one node and one graph.
    This function does not support jax.jit, because the shape of the output
    is data-dependent.
    Args:
    graph: ``GraphsTuple`` padded with dummy graph and empty graphs.
    n_node: the number of nodes in the padded ``GraphsTuple``.
    n_edge: the number of edges in the padded ``GraphsTuple``.
    n_graph: the number of graphs in the padded ``GraphsTuple``. Default is 2,
      which is the lowest possible value, because we always have at least one
      graph in the original ``GraphsTuple`` and we need one dummy graph for the
      padding.
    Raises:
    ValueError: if the passed ``n_graph`` is smaller than 2.
    RuntimeError: if the given ``GraphsTuple`` is too large for the given
      padding.
    Returns:
    A padded ``GraphsTuple``.
    """
    np = jnp
    if n_graph < 2:
        raise ValueError(
            f'n_graph is {n_graph}, which is smaller than minimum value of 2.')
    graph = jax.device_get(graph)
    pad_n_node = int(n_node - np.sum(graph.n_node))
    pad_n_edge = int(n_edge - np.sum(graph.n_edge))
    pad_n_graph = int(n_graph - graph.n_node.shape[0])
    if pad_n_node <= 0 or pad_n_edge < 0 or pad_n_graph <= 0:
        raise RuntimeError(
            'Given graph is too large for the given padding. difference: '
            f'n_node {pad_n_node}, n_edge {pad_n_edge}, n_graph {pad_n_graph}')

    pad_n_empty_graph = pad_n_graph - 1

    tree_nodes_pad = (
        lambda leaf: np.zeros((pad_n_node,) + leaf.shape[1:], dtype=leaf.dtype))
    tree_edges_pad = (
        lambda leaf: np.zeros((pad_n_edge,) + leaf.shape[1:], dtype=leaf.dtype))
    tree_globs_pad = (
        lambda leaf: np.zeros((pad_n_graph,) + leaf.shape[1:], dtype=leaf.dtype))

    padding_graph = GraphsTuple(
        n_node=np.concatenate(
            [np.array([pad_n_node], dtype=np.int64),
             np.zeros(pad_n_empty_graph, dtype=np.int64)]),
        n_edge=np.concatenate(
            [np.array([pad_n_edge], dtype=np.int64),
             np.zeros(pad_n_empty_graph, dtype=np.int64)]),
        nodes=jax.tree_map(tree_nodes_pad, graph.nodes),
        edges=jax.tree_map(tree_edges_pad, graph.edges),
        globals=jax.tree_map(tree_globs_pad, graph.globals),
        senders=np.zeros(pad_n_edge, dtype=np.int64),
        receivers=np.zeros(pad_n_edge, dtype=np.int64),
        e_order=jax.tree_map(tree_edges_pad, graph.e_order),
        e_mask=jax.tree_map(tree_edges_pad, graph.e_mask),
        n_mask=jax.tree_map(tree_nodes_pad, graph.n_mask),
    )
    return _batch([graph, padding_graph], np_=np)
