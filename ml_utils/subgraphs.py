# -*- Encoding: UTF-8 -*-

import torch
import networkx as nx

def get_edge_index(node_to: int, node_from: int, edge_index: torch.tensor) -> int:
    """Returns the column in edge_index corresponding to the edge (node_to, node_from)."""
    idx_to = edge_index[0, :] == node_to
    idx_fr = edge_index[1, :] == node_from

    idx_abs = torch.nonzero(idx_to & idx_fr).squeeze().item()

    return(idx_abs)


def get_neighbors(G, root_node: int, excl_list: list, depth: int, nb_dict: dict, maxdepth:int) -> dict:
    """Recursively build subgraph of nodes from depth k to k+1. Starting at
    depth 0, up to maxdepth."""
    neighbor_set_k = [n for n in G[root_node] if n not in excl_list]
    try:
        nb_dict[maxdepth + 1 - depth].append((root_node, neighbor_set_k))
    except:
        nb_dict[maxdepth + 1 - depth] = [(root_node, neighbor_set_k)]

    if depth > 0:
        for n in neighbor_set_k:
            get_neighbors(G, n, excl_list + [root_node], depth - 1, nb_dict, maxdepth)
    else:
        return nb_dict

def kneighbor_squashed(root_vtx: int, conns_3d: dict, neighbors: set, k: int=2):
    """Generates a 'squashed' neighborhood of root_vtx where all nodes that can
    be reached through by max k edges.
    """

    assert((k >= 1) & (k <= 5))
    if k == 1:
        # This terminates the recursion. Add only the nearest neighbors of the root vertex
        # to the neighbors set
        for vtx in conns_3d[root_vtx]:
            neighbors.add(vtx)
    else:
        for vtx in conns_3d[root_vtx]:
            neighbors = kneighbor_squashed(vtx, conns_3d, neighbors, k - 1)

    return neighbors



# Code below builds directed subgraphs from k=2 nodes to k=1 nodes.
# The layout of edge_index is target_to_source, with targets in row 0 and
# sources in row 1.
# Input is the nb_dict structure, returned from get_neighbors.

def build_subgraph(edge_index: torch.tensor, weight: torch.tensor, depth: int=2) -> list:
    """Constructs subgraph from nodes at depth to depth+1.

    Parameters:
    -----------
    edge_index: pytorch-geometric edge_index structure
    weight: corresponds to pytorch-geometric edge_attr
    depth: distance to the root vertex 0

    Returns:
    --------
    list of (edge_index: torch.tensor, weigh: torch.tensor) which define the subgraph
    """

    # Construct an networkX graph from the current edge_index
    my_G = nx.Graph()
    for i, j in zip(edge_index[0,:], edge_index[1,:]):
        my_G.add_edge(i.item(), j.item())

    # Recursively find connections from the root_vertex (0), up to vertices
    # 2 edges away. Store results in nb_dict
    nb_dict = {}
    get_neighbors(my_G, 0, [], depth, nb_dict, depth)

    # Construct an edge_index and weight structure of the vertices 2 nodes away
    # from the root vertex to the vertex 1 node away. Keep the weight of the connections.

    # Get the total number of edges for k=2 nodes from the dictionary
    num_edges = 0
    for g in nb_dict[depth]:
        num_edges += len(g[1])

    # Define empty edge_index and weight tensors that will hold the connections
    ei_sub = torch.zeros([2, num_edges], dtype=torch.long)
    wt_sub = torch.zeros([num_edges, 3])

    edge_ctr = 0

    for g in nb_dict[depth]:
        cur_edges = len(g[1])
        for idx_e in range(edge_ctr, edge_ctr + cur_edges):
            ei_sub[0, idx_e] = g[0]
            ei_sub[1, idx_e] = g[1][idx_e - edge_ctr]

            widx = get_edge_index(g[0], g[1][idx_e - edge_ctr], edge_index)
            wt_sub[idx_e, :] = weight[widx, :]

        edge_ctr += cur_edges

    return ei_sub, wt_sub



def build_subgraphs_lvl1(G):
    neighbor_set_0 = G[0]
    edge_index_1_list = []
    weights_1_list = []

    # Build subgraphs for every node in the 1-neighborhood of the root node
    for k in neighbor_set_0:
        neighbor_set_1 =  [k2 for k2 in G[k] if k2 not in [0]]
        num_edges = len(neighbor_set_1)

        # Build an edge index and copy the features and weights from the original graph
        ei_sub = torch.zeros([2, num_edges], dtype=torch.long)
        wt_sub = torch.zeros([num_edges, 3], dtype=torch.float64)


        ei_sub[0, :] = k
        for ll in range(num_edges):
            ei_sub[1, ll] = neighbor_set_1[ll]
            widx = get_edge_index(k, neighbor_set_1[ll], edge_index)
            wt_sub[ll, :] = weight[widx, :]

        edge_index_1_list.append(ei_sub)
        weights_1_list.append(wt_sub)

        #print(neighbor_set_1, ei_sub, wt_sub)
        #subgraph_list.append(Data(x=graph_list[gidx].x, edge_index=ei_sub, weight=wt_sub))

    #print(edge_index_1_list)
    #print(weights_1_list)
    ei_lvl1 = torch.cat(edge_index_1_list, dim=1)
    wt_lvl1 = torch.cat(weights_1_list, dim=0)

    subgraph_lvl1 = Data(x=graph_list[gidx].x, edge_index=ei_lvl1, weight=wt_lvl1)

    return(subgraph_lvl1)
