# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 20:16:32 2021

@author: Defryder
"""

import networkx as nx
import itertools
from networkx.convert_matrix import _generate_weighted_edges
from networkx.algorithms import bipartite
import numpy as np
import scipy
from scipy import sparse


def biadjacency_matrix(
    G, row_order, column_order=None, dtype=None, weight="weight", format="csr"
):
    r"""Returns the biadjacency matrix of the bipartite graph G.

    Let `G = (U, V, E)` be a bipartite graph with node sets
    `U = u_{1},...,u_{r}` and `V = v_{1},...,v_{s}`. The biadjacency
    matrix [1]_ is the `r` x `s` matrix `B` in which `b_{i,j} = 1`
    if, and only if, `(u_i, v_j) \in E`. If the parameter `weight` is
    not `None` and matches the name of an edge attribute, its value is
    used instead of 1.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    row_order : list of nodes
       The rows of the matrix are ordered according to the list of nodes.

    column_order : list, optional
       The columns of the matrix are ordered according to the list of nodes.
       If column_order is None, then the ordering of columns is arbitrary.

    dtype : NumPy data-type, optional
        A valid NumPy dtype used to initialize the array. If None, then the
        NumPy default is used.

    weight : string or None, optional (default='weight')
       The edge data key used to provide each value in the matrix.
       If None, then each edge has weight 1.

    format : str in {'bsr', 'csr', 'csc', 'coo', 'lil', 'dia', 'dok'}
        The type of the matrix to be returned (default 'csr').  For
        some algorithms different implementations of sparse matrices
        can perform better.  See [2]_ for details.

    Returns
    -------
    M : SciPy sparse matrix
        Biadjacency matrix representation of the bipartite graph G.

    Notes
    -----
    No attempt is made to check that the input graph is bipartite.

    For directed bipartite graphs only successors are considered as neighbors.
    To obtain an adjacency matrix with ones (or weight values) for both
    predecessors and successors you have to generate two biadjacency matrices
    where the rows of one of them are the columns of the other, and then add
    one to the transpose of the other.

    See Also
    --------
    adjacency_matrix
    from_biadjacency_matrix

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Adjacency_matrix#Adjacency_matrix_of_a_bipartite_graph
    .. [2] Scipy Dev. References, "Sparse Matrices",
       https://docs.scipy.org/doc/scipy/reference/sparse.html
    """
    from scipy import sparse

    nlen = len(row_order)
    if nlen == 0:
        raise nx.NetworkXError("row_order is empty list")
    if len(row_order) != len(set(row_order)):
        msg = "Ambiguous ordering: `row_order` contained duplicates."
        raise nx.NetworkXError(msg)
    if column_order is None:
        column_order = list(set(G) - set(row_order))
    mlen = len(column_order)
    if len(column_order) != len(set(column_order)):
        msg = "Ambiguous ordering: `column_order` contained duplicates."
        raise nx.NetworkXError(msg)

    row_index = dict(zip(row_order, itertools.count()))
    col_index = dict(zip(column_order, itertools.count()))

    if G.number_of_edges() == 0:
        row, col, data = [], [], []
    else:
        row, col, data = zip(
            *(
                (row_index[u], col_index[v], d.get(weight, 1))
                for u, v, d in G.edges(row_order, data=True)
                if u in row_index and v in col_index
            )
        )
    M = sparse.coo_matrix((data, (row, col)), shape=(nlen, mlen), dtype=dtype)
    try:
        return M.asformat(format)
    # From Scipy 1.1.0, asformat will throw a ValueError instead of an
    # AttributeError if the format if not recognized.
    except (AttributeError, ValueError) as e:
        raise nx.NetworkXError(f"Unknown sparse matrix format: {format}") from e

def from_biadjacency_matrix(A, create_using=None, edge_attribute="weight"):
    r"""Creates a new bipartite graph from a biadjacency matrix given as a
    SciPy sparse matrix.

    Parameters
    ----------
    A: scipy sparse matrix
      A biadjacency matrix representation of a graph

    create_using: NetworkX graph
       Use specified graph for result.  The default is Graph()

    edge_attribute: string
       Name of edge attribute to store matrix numeric value. The data will
       have the same type as the matrix entry (int, float, (real,imag)).

    Notes
    -----
    The nodes are labeled with the attribute `bipartite` set to an integer
    0 or 1 representing membership in part 0 or part 1 of the bipartite graph.

    If `create_using` is an instance of :class:`networkx.MultiGraph` or
    :class:`networkx.MultiDiGraph` and the entries of `A` are of
    type :class:`int`, then this function returns a multigraph (of the same
    type as `create_using`) with parallel edges. In this case, `edge_attribute`
    will be ignored.

    See Also
    --------
    biadjacency_matrix
    from_numpy_array

    References
    ----------
    [1] https://en.wikipedia.org/wiki/Adjacency_matrix#Adjacency_matrix_of_a_bipartite_graph
    """
    G = nx.empty_graph(0, create_using)
    n, m = A.shape
    # Make sure we get even the isolated nodes of the graph.
    G.add_nodes_from(range(n), bipartite=0)
    G.add_nodes_from(range(n, n + m), bipartite=1)
    # Create an iterable over (u, v, w) triples and for each triple, add an
    # edge from u to v with weight w.
    triples = ((u, n + v, d) for (u, v, d) in _generate_weighted_edges(A))
    # If the entries in the adjacency matrix are integers and the graph is a
    # multigraph, then create parallel edges, each with weight 1, for each
    # entry in the adjacency matrix. Otherwise, create one edge for each
    # positive entry in the adjacency matrix and set the weight of that edge to
    # be the entry in the matrix.
    if A.dtype.kind in ("i", "u") and G.is_multigraph():
        chain = itertools.chain.from_iterable
        triples = chain(((u, v, 1) for d in range(w)) for (u, v, w) in triples)
    G.add_weighted_edges_from(triples, weight=edge_attribute)
    return G

def minimum_weight_full_matching(G, top_nodes=None, weight="weight"):
    r"""Returns a minimum weight full matching of the bipartite graph `G`.

    Let :math:`G = ((U, V), E)` be a weighted bipartite graph with real weights
    :math:`w : E \to \mathbb{R}`. This function then produces a matching
    :math:`M \subseteq E` with cardinality

    .. math::
       \lvert M \rvert = \min(\lvert U \rvert, \lvert V \rvert),

    which minimizes the sum of the weights of the edges included in the
    matching, :math:`\sum_{e \in M} w(e)`, or raises an error if no such
    matching exists.

    When :math:`\lvert U \rvert = \lvert V \rvert`, this is commonly
    referred to as a perfect matching; here, since we allow
    :math:`\lvert U \rvert` and :math:`\lvert V \rvert` to differ, we
    follow Karp [1]_ and refer to the matching as *full*.

    Parameters
    ----------
    G : NetworkX graph

      Undirected bipartite graph

    top_nodes : container

      Container with all nodes in one bipartite node set. If not supplied
      it will be computed.

    weight : string, optional (default='weight')

       The edge data key used to provide each value in the matrix.

    Returns
    -------
    matches : dictionary

      The matching is returned as a dictionary, `matches`, such that
      ``matches[v] == w`` if node `v` is matched to node `w`. Unmatched
      nodes do not occur as a key in `matches`.

    Raises
    ------
    ValueError
      Raised if no full matching exists.

    ImportError
      Raised if SciPy is not available.

    Notes
    -----
    The problem of determining a minimum weight full matching is also known as
    the rectangular linear assignment problem. This implementation defers the
    calculation of the assignment to SciPy.

    References
    ----------
    .. [1] Richard Manning Karp:
       An algorithm to Solve the m x n Assignment Problem in Expected Time
       O(mn log n).
       Networks, 10(2):143–152, 1980.

    """
    try:
        import numpy as np
        import scipy.optimize
    except ImportError as e:
        raise ImportError(
            "minimum_weight_full_matching requires SciPy: " + "https://scipy.org/"
        ) from e
    left, right = nx.bipartite.sets(G, top_nodes)
    U = list(left)
    V = list(right)
    # We explicitly create the biadjancency matrix having infinities
    # where edges are missing (as opposed to zeros, which is what one would
    # get by using toarray on the sparse matrix).
    weights_sparse = biadjacency_matrix(G, row_order=U, column_order=V, weight=weight, format="coo")
    weights = np.full(weights_sparse.shape, np.inf)
    weights[weights_sparse.row, weights_sparse.col] = weights_sparse.data
    left_matches = scipy.optimize.linear_sum_assignment(weights)
    d = {U[u]: V[v] for u, v in zip(*left_matches)}
    # d will contain the matching from edges in left to right; we need to
    # add the ones from right to left as well.
    d.update({v: u for u, v in d.items()})
    return d

def generate_biparite_s(x, M, N, K, c, p, d, b, env):
    
    if np.all([[not (j%1) for j in i]for i in x]):
        return x
    
    k = []
    k_inv = []
    count = 0
    for i in range(M):
        k.append(int(np.ceil(np.sum(x[i]))))
        for j in range(k[i]):
            k_inv.append(count)
        count = count + 1
    
    subM = int(np.sum(k))

    bip = np.zeros((subM, N))
    
    B = nx.Graph()
    B.add_nodes_from(range(subM), bipartite=0)
    B.add_nodes_from(range(subM, subM + N), bipartite=1)
    
    for i in range(M):
        
        subi = int(sum(k[:i]))
        
        ordered_pi = sorted([[(p[i][j])*np.ceil(x[i][j]), j] for j in range(N)], reverse=True, key=lambda x: x[0])
        
        count = 0
        e = ordered_pi[count]
        
        offset = 0
        
        while count <= len(ordered_pi)-1 and ordered_pi[count][0] != 0:
            e = ordered_pi[count]
            filler = 0
            if np.sum(bip[subi + offset]) + x[i][e[1]] >= 1:
                filler = 1 - np.sum(bip[subi + offset])
#                bip[subi + offset - 1][e[1]] = filler
#                B.add_edge(subi + offset, subM + e[1], weight = filler)
                bip[subi + offset][e[1]] = filler
                B.add_edge(subi + offset, subM + e[1], weight = (c[i][e[1]])*x[i][e[1]])
                offset = offset + 1
            
            if x[i][e[1]] - filler > 0.001:
                bip[subi + offset][e[1]] = bip[subi + offset][e[1]] + x[i][e[1]] - filler
                B.add_edge(subi + offset, subM + e[1], weight = (c[i][e[1]])*x[i][e[1]])
            
            count = count + 1
    
    to_remove = [(a,b) for a, b, attrs in B.edges(data=True) if attrs["weight"] <= 0.00001]
    B.remove_edges_from(to_remove)
    
    B.remove_nodes_from(list(nx.isolates(B)))
    
    top_nodes = {n for n, d in B.nodes(data=True) if d["bipartite"] == 1}
    
    match = minimum_weight_full_matching(B, top_nodes)
    
    out = np.zeros((M, N))
    
    for i, m in enumerate(k_inv):
        try:
            t = match[i] - subM
            out[m][t] = 1
        except:
            pass
    
    return out

def generate_biparite(x, M, N, K, c, p, d, b, env):
    
    if np.all([[not (j%1) for j in i]for i in x]):
        e = np.zeros((M, K))
        for m in range(M):
            for t in range(N):
                if x[m][t] == 1 and e[m][env[t]] == 0:
                    e[m][env[t]] = 1
        
        
        return x, e
    
    k = []
    k_inv = []
    count = 0
    for i in range(M):
        k.append(int(np.ceil(np.sum(x[i]))))
        for j in range(k[i]):
            k_inv.append(count)
        count = count + 1
    
    subM = int(np.sum(k))

    bip = np.zeros((subM, N))
    
    B = nx.Graph()
    B.add_nodes_from(range(subM), bipartite=0)
    B.add_nodes_from(range(subM, subM + N), bipartite=1)
    
    for i in range(M):
        
        subi = int(sum(k[:i]))
        
        ordered_pi = sorted([[(p[i][j]+b[i][env[j]])*np.ceil(x[i][j]), j] for j in range(N)], reverse=True, key=lambda x: x[0])
        
        count = 0
        e = ordered_pi[count]
        
        offset = 0
        
        while count <= len(ordered_pi)-1 and ordered_pi[count][0] != 0:
            e = ordered_pi[count]
            filler = 0
            if np.sum(bip[subi + offset]) + x[i][e[1]] >= 1:
                filler = 1 - np.sum(bip[subi + offset])
#                bip[subi + offset - 1][e[1]] = filler
#                B.add_edge(subi + offset, subM + e[1], weight = filler)
                bip[subi + offset][e[1]] = filler
                B.add_edge(subi + offset, subM + e[1], weight = (c[i][e[1]]+d[i][env[e[1]]])*x[i][e[1]])
                offset = offset + 1
            
            if x[i][e[1]] - filler > 0.001:
                bip[subi + offset][e[1]] = bip[subi + offset][e[1]] + x[i][e[1]] - filler
                B.add_edge(subi + offset, subM + e[1], weight = (c[i][e[1]]+d[i][env[e[1]]])*x[i][e[1]])
            
            count = count + 1
    
    to_remove = [(a,b) for a, b, attrs in B.edges(data=True) if attrs["weight"] <= 0.00001]
    B.remove_edges_from(to_remove)
    
    B.remove_nodes_from(list(nx.isolates(B)))
    
    top_nodes = {n for n, d in B.nodes(data=True) if d["bipartite"] == 1}
    
    match = minimum_weight_full_matching(B, top_nodes)
    
    out = np.zeros((M, N))
    out_e = np.zeros((M, K))
    
    for i, m in enumerate(k_inv):
        try:
            t = match[i] - subM
            out[m][t] = 1
            if out_e[m][env[t]] == 0:
                out_e[m][env[t]] = 1
        except:
            pass
    
    
    
    return out, out_e
    