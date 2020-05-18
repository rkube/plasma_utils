# File ml_utils/pyg_xgc_loaders.py
# -*- Encoding: UTF-8 -*-

"""
Define custom data loaders for xgc data.
"""

import sys
from os.path import join
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric import utils

import networkx as nx

from sklearn import preprocessing
from itertools import product

sys.path.append("/home/rkube/software/adios2-release_25/lib64/python3.7/site-packages")
import adios2
from ..xgc_utils import xgc_grid, xgc_helpers




def build_graph_weights(data_dir, weights_scaler, dt=1e-8, num_planes=8):
    """Builds the edge_index and edge_weights structures for the XGC simulation in datadir

    Parameters:
    -----------
    datadir, string: Directory where the xgc.mesh and xgc.bfield files are.
    weights_scaler, callable: Scaling function applied to weights.
    num_planes, int: Number of poloidal planes
    dt, float: Time step size
    """
    with np.load(join(data_dir, "xgc.mesh.npz")) as df:
        coords = df["/coordinates/values"]
        nextnode = df["nextnode"]
        nc = df["/cell_set[0]/node_connect_list"]

    with np.load(join(data_dir, "xgc.bfield.npz")) as df:
        Bvec = df["/node_data[0]/values"]

    # Calculate the total magnetic field
    Btotal = np.sqrt(np.sum(Bvec**2.0, axis=1))  # Total magnetic field, in T
    # Get background density and temperature. Hard-code those for test cases.
    ne0 = 1e19 # Background electron density, in m^-3
    ni0 = 1e19 # Background ion density, in m^-3
    Te0 = 2e3 # Background electron temperaure, in eV
    Ti0 = 2e3 # Background ion temperature, in eV

    print(f"Using hard-coded value for ne0: {ne0}m^-3")
    print(f"Using hard-coded value for ni0: {ni0}m^-3")
    print(f"Using hard-coded value for Te0: {Te0}eV")
    print(f"Using hard-coded value for Ti0: {Ti0}eV")

    mu0 = 4. * np.pi * 1e-7 # Vacuum permeability, in H/m
    eps0 = 8.854e-12 # Vacuum permittivity, in F/m
    e = 1.602e-19 # Elementary charge, in C
    me = 9.109e-31 # Electron mass, in kg
    mi = 1.67e-27 # Proton mass, in kg
    c0 = 3e8 # Speed of light in m/s

    VA = Btotal / np.sqrt(mu0 * mi * ni0)
    VS = np.sqrt(e * Te0 / me)
    wpe = np.sqrt(ne0 * e * e / me / eps0)
    de = c0 / wpe

    print(f"mean(VA) = {VA.mean():4.2e}m/s")
    print(f"VS = {VS:4.2e}m/s")
    print(f"omega_pe = {wpe:4.2e}Hz")
    print(f"Skin depth de = {de:f}m")

    # Set parallel and perpendicular normalizations
    norm_par = 0.5 * (VA + VS) * dt
    norm_perp = de
    rbf = lambda x: 1./np.sqrt(1. + x*x)

    # Create the mesh and get the node connections
    mesh = xgc_grid.xgc_mesh(nc, coords)
    conns = xgc_grid.get_node_connection(nc, nextnode, local_plane=False)

    nodes_per_plane = max(conns.keys()) + 1
    num_vertices = nodes_per_plane * num_planes
    print(f"Nodes per plane: {nodes_per_plane}, poloidal planes: {num_planes}")

    # Calculate connection weights based on cartesian distance an RBFs
    print(f"Calculating weights")
    print(f"Using the old version: normalized_cartesian_distance_old")
    all_weights_3 = []
    for from_node in conns.keys():
        dist_normed = xgc_grid.normalized_cartesian_distance_old(from_node, conns[from_node], coords, norm_par, norm_perp)
        all_weights_3.append(dist_normed)

    all_weights_3 = np.vstack(all_weights_3)
    all_weights_3 = weights_scaler(all_weights_3)

    print(f"Scaling weights")
    #scaler_r = preprocessing.MinMaxScaler()
    #all_weights_3 = scaler_r.fit_transform(all_weights_3)

    # Generate the weights tensor. Repeat the weights from the stencil of the first plane
    weights = torch.tensor(all_weights_3)
    weights = weights.repeat((num_planes, 1)).T
    print(f"Weights repeated over {num_planes} shape = ", weights.shape)


    print(f"Calculating connections in the mesh")
    all_conns = []
    for plane in range(num_planes):
        for from_node in conns.keys():
            rv = xgc_grid.shift_connections(from_node, conns[from_node], nodes_per_plane, plane, num_planes)
            all_conns += rv

    print(f"Calculating global edge index")
    edge_index_all = torch.tensor(all_conns).contiguous().T

    return edge_index_all, weights, Btotal






def build_graph_list(data_x, data_y, edge_index_g, weights_g, cuda=True):
    """Constructs the list of star-shaped sub-graphs

    Parameters:
    -----------
    data_x, torch.tensor: vertex features with shape [nnodes, nfeatures]
    data_y, torch.tensor: target features with shape [nnodes, nfeatures]
    edge_index_g, torch.tensor: edge indices for the global graph. shape [2, num_edges]. max(edge_index_g) == nnodes -1
    weights_g, torch.tensor: connection weights of the global graph. shape [dim_weights, num_edges]
    cuda, bool: If True, process everything on cuda (recommended). If not, use the cpu (not recommended).

    Returns:
    --------
    graph_list: List of star-shaped graphs.

    """
    torch.set_default_dtype(torch.float64)

    if cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    data_x.to(device)
    data_y.to(device)
    edge_index_g.to(device)
    weights_g.to(device)

    num_vertices = edge_index_g.max().item() + 1
    graph_list = []

    for vertex in range(num_vertices):
        # 1. Find all edges connecting to the current vertex. These will determine the current sub-graph
        # We extracting a star-shaped sub-graph with the current vertex in the center. That is, all connections
        # are to be unique.

        # Extract all edges connecting to the current vertex
        target, _ = edge_index_g
        subgraph_edge_indices = target == vertex

        # Get the vertices connecting to the current vertex
        _, vertex_idx = edge_index_g[:, subgraph_edge_indices]

        # Test if there are zero-values in the graph. If there are, skip this one.
        # If we are working with residuals we should make sure that we don't discard
        # a graph just because they are small. Here we have them in the last two indices,
        # therefore use :-2 slicing in the second dimension.
        if (((data_x[vertex_idx, :-2].abs() < 1e-16).sum() > 0) |
            (torch.isnan(data_x[vertex_idx, :]).sum() > 0)):
            #print(f"Skipping graph with zero-values node features: vertex={vertex}")
            continue

        # Forthcoming, we assume that there are no duplicate connections.
        assert(torch.unique(vertex_idx).size()[0] == vertex_idx.size()[0])
        # Extract the edge weights relevant to the current sub-graph from the global edge weights
        weights_subgraph = weights_g[:, subgraph_edge_indices]

        # The number of nodes in the current subgraph is just the length
        nnodes = vertex_idx.shape[0]

        # Next: Swap the first item with the current vertex number in the subgraph
        if vertex_idx[0] != vertex:
            # Find index of the current vertex and switch it with the first item
            switch_idx = (vertex_idx == vertex).nonzero().item()

            # Switch weights
            _weight_tmp = weights_subgraph[:, 0].clone().detach()
            weights_subgraph[:, 0] = weights_subgraph[:, switch_idx]
            weights_subgraph[:, switch_idx] = _weight_tmp

            # Switch vertex numbers
            vertex_idx[switch_idx] = vertex_idx[0]
            vertex_idx[0] = vertex

        # Set weights for the self-loop to unity
        weights_subgraph[:, 0] = torch.tensor([1.0, 1.0, 1.0])

        # Build a new edge_index, where the current vertex, indexed by zero, is connected
        # to all other vertices. This is a directed graph

        # OBS! Ordering edge_index in this way [[0, 0, ..., 0], [0, 1, .., nnodes]]
        # requires to set flow="target_to_source" in MessagePassing classes
        edge_index_zero = torch.stack([torch.zeros(nnodes, dtype=torch.long),
                                    torch.arange(nnodes, dtype=torch.long)])

        graph_list.append(Data(x=data_x[vertex_idx, :],
                               y=data_y[vertex, :],
                               edge_index=edge_index_zero,
                               edge_attr=weights_subgraph.T,
                               num_nodes=nnodes,
                               original_vertex=vertex))

    return graph_list


def collate(data_list):
    r"""Collates a python list of data objects to the internal storage
    format of :class:`torch_geometric.data.InMemoryDataset`.

    Code copied from torch_geometric.data.InMemoryDataset"""

    keys = data_list[0].keys
    data = data_list[0].__class__()

    for key in keys:
        data[key] = []
    slices = {key: [0] for key in keys}

    for item, key in product(data_list, keys):
        data[key].append(item[key])
        if torch.is_tensor(item[key]):
            s = slices[key][-1] + item[key].size(
                item.__cat_dim__(key, item[key]))
        else:
            s = slices[key][-1] + 1
        slices[key].append(s)

    if hasattr(data_list[0], '__num_nodes__'):
        data.__num_nodes__ = []
        for item in data_list:
            data.__num_nodes__.append(item.num_nodes)

    for key in keys:
        item = data_list[0][key]
        if torch.is_tensor(item):
            data[key] = torch.cat(data[key],
                                    dim=data.__cat_dim__(key, item))
        elif isinstance(item, int) or isinstance(item, float):
            data[key] = torch.tensor(data[key])

        slices[key] = torch.tensor(slices[key], dtype=torch.long)

    return data, slices



class XGCGraphDataset(InMemoryDataset):
    r"""
    Args:
        root(string): Root directory where the bp files reside. The processed dataset will be saved in
            the same directory.
    """
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, tidx=4):
        self.tidx = tidx
        super(InMemoryDataset, self).__init__(root, transform, pre_transform, pre_filter)

        print("processed_paths = ", self.processed_paths)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        print("raw_file_name called")
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        print("processed_file_names called")
        return [f"data_tidx{self.tidx:05d}.pt"]

    def _download(self):
        pass

    def process(self):
        torch.set_default_dtype(torch.float64)
        cuda = torch.device('cuda')

        print("Processing...")

        with adios2.open(join(self.root, "xgc.mesh.bp"), "r") as df:
            coords = df.read("/coordinates/values")
            nextnode = df.read("nextnode")
            nc = df.read("/cell_set[0]/node_connect_list")
            df.close()

        with adios2.open(join(self.root, "xgc.bfield.bp"), "r") as df:
            Bvec = df.read("/node_data[0]/values")
            v2 = df.read("/node_data[1]/values")
            df.close()

        # Calculate the total magnetic field
        Btotal = np.sqrt(np.sum(Bvec**2.0, axis=1))  # Total magnetic field, in T
        # Get background density and temperature. Hard-code those for test cases.
        ne0 = 1e19 # Background electron density, in m^-3
        ni0 = 1e19 # Background ion density, in m^-3
        Te0 = 2e3 # Background electron temperaure, in eV
        Ti0 = 2e3 # Background ion temperature, in eV

        print(f"Using hard-coded value for ne0: {ne0}m^-3")
        print(f"Using hard-coded value for ni0: {ni0}m^-3")
        print(f"Using hard-coded value for Te0: {Te0}eV")
        print(f"Using hard-coded value for Ti0: {Ti0}eV")

        mu0 = 4. * np.pi * 1e-7 # Vacuum permeability, in H/m
        eps0 = 8.854e-12 # Vacuum permittivity, in F/m
        e = 1.602e-19 # Elementary charge, in C
        me = 9.109e-31 # Electron mass, in kg
        mi = 1.67e-27 # Proton mass, in kg
        c0 = 3e8 # Speed of light in m/s

        VA = Btotal / np.sqrt(mu0 * mi * ni0)
        VS = np.sqrt(e * Te0 / me)
        wpe = np.sqrt(ne0 * e * e / me / eps0)
        de = c0 / wpe

        print(f"mean(VA) = {VA.mean():4.2e}m/s")
        print(f"VS = {VS:4.2e}m/s")
        print(f"omega_pe = {wpe:4.2e}Hz")
        print(f"Skin depth de = {de:f}m")

        # Set parallel and perpendicular normalizations
        dt = 1e-8
        norm_par = 0.5 * (VA + VS) * dt
        norm_perp = de
        rbf = lambda x: 1./np.sqrt(1. + x*x)

        # Create the mesh and get the node connections
        mesh = xgc_grid.xgc_mesh(nc, coords)
        conns = xgc_grid.get_node_connection(nc, nextnode, local_plane=False)

        nodes_per_plane = max(conns.keys()) + 1
        num_planes = 8
        print(f"Nodes per plane: {nodes_per_plane}, poloidal planes: {num_planes}")


        # Calculate connection weights based on cartesian distance an RBFs
        print(f"Calculating weights")
        all_weights_3 = []
        for from_node in conns.keys():
            dist_normed = xgc_grid.normalized_cartesian_distance(from_node, conns[from_node], coords, norm_par, norm_perp)
            all_weights_3.append(dist_normed)

        all_weights_3 = np.vstack(all_weights_3)

        print(f"Scaling weights")
        scaler_r = preprocessing.MinMaxScaler()
        all_weights_3 = scaler_r.fit_transform(all_weights_3)

        # Generate the weights tensor. Repeat the weights from the stencil of the first plane
        weights = torch.tensor(all_weights_3)
        weights = weights.repeat((num_planes, 1)).T
        print("Weights repeated over {num_planes} shape = ", weights.shape)

        print(f"Calculating connections in the mesh")
        all_conns = []
        for plane in range(num_planes):
            for from_node in conns.keys():
                rv = xgc_grid.shift_connections(from_node, conns[from_node], nodes_per_plane, plane, num_planes)
                all_conns += rv

        print(f"Calculating global edge index")
        edge_index_all = torch.tensor(all_conns).contiguous().T

        # Compile feature vectors on all nodes
        print(f"Compiling node features")
        node_features = {}
        with adios2.open(join(self.root, f"xgc.3d.0000{self.tidx}.c.bp"), "r") as df:
            for key in ["eden", "iden", "u_e", "u_i", "dpot", "a_par"]:
                node_features[key] = df.read(key, start=[0, 0], count=[nodes_per_plane, num_planes], step_start=0, step_count=1)

        for key in node_features.keys():
            node_features[key] = node_features[key][0, :, :].T.flatten()
        node_features["B"] = np.tile(Btotal, (num_planes, 1)).T.flatten()

        # Identify vanishingly small features
        _zero_idx = np.zeros([list(node_features.values())[0].shape[0], len(node_features.keys())], dtype=np.bool)

        for idx, data in enumerate(node_features.values()):
            _zero_idx[:, idx] = abs(data) < 1e-16
        zero_idx = np.argwhere(_zero_idx.any(axis=1)).ravel()

        # Create a boolean index array which indices only positive values in the fields
        nzero_idx_bool = np.ones(list(node_features.values())[0].shape[0], dtype=np.bool)
        nzero_idx_bool[zero_idx] = False
        nzero_vertices = (nzero_idx_bool).nonzero()[0]

        print(f"Zero_idx: {zero_idx.shape[0]:d}/{list(node_features.values())[0].shape[0]:d} elements below threshold")
        print(f"Identified {nzero_vertices.size} vertices with non-zero feature vectors")

        # Scale features
        _res = [(val - val[nzero_idx_bool].mean()) / val[nzero_idx_bool].std() for val in node_features.values()]
        data_x = torch.tensor(np.vstack(_res).T)
        _res = 0.0
        data_x_g = data_x.to(device=cuda)
        print(f"Created scaled feature vector. shape: {data_x.shape}")

        # Create target vector
        # Compile target features
        target_features = {}

        with adios2.open(join(self.root, f"xgc.3d.0000{self.tidx + 1}.c.bp"), "r") as df:
            for key in ["dpot", "a_par"]:
                target_features[key] = df.read(key, start=[0, 0], count=[nodes_per_plane, num_planes],
                                            step_start=0, step_count=1).T.flatten()

        _res = [(val - trg[nzero_idx_bool].mean()) / trg[nzero_idx_bool].std()
                    for val, trg in zip(target_features.values(), [node_features[k] for k in ["dpot", "a_par"]])]

        data_y = torch.tensor(np.vstack(_res).T)
        data_y_g = data_y.to(device=cuda)
        _res = 0.0
        print(f"Created scaled target vector. shape: {data_y.shape}")


        # Identify vertices with vanishing feature vectors.
        # This section is slow and should be done on device
        zero_idx_g = torch.from_numpy(zero_idx).cuda()
        edge_index_g = edge_index_all.to(device=cuda)
        weights_g = weights.to(device=cuda)

        zi_next_g = torch.zeros(edge_index_all.shape[1], dtype=torch.bool, device=cuda)

        for zi in zero_idx_g:
            # Mark locations of current zero-node in edge_index_g as True
            zi_current_g = (edge_index_g == zi).any(0)
            # Update locations in zi_next_g
            zi_next_g = zi_next_g | zi_current_g

        zi_next = zi_next_g.cpu()

        print(f"Indexing {zi_next_g.sum():d} vertices to zero")
        # Gives the list of good vertices

        ### Create a list of subgraphs, with each non-zero vertex in the middle.

        ### In each sub-graph, the non-zero vertex is indexed by zero. The vertices it connects to are re-indexed starting
        ### from 1. The original vertex indices are used to compile the node features and the regression targets.

        graph_list = []

        for vertex in nzero_vertices:
            # 1. Find all edges connecting to the current vertex. These will determine the current sub-graph
            # We extracting a star-shaped sub-graph with the current vertex in the center. That is, all connections
            # are to be unique.

            # Extract all edges connecting to the current vertex
            target, _ = edge_index_g
            subgraph_edge_indices = target == vertex

            # Get the vertices connecting to the current vertex
            _, vertex_idx = edge_index_g[:, subgraph_edge_indices]

            # Forthcoming, we assume that there are no duplicate connections.
            assert(torch.unique(vertex_idx).size()[0] == vertex_idx.size()[0])
            # Extract the edge weights relevant to the current sub-graph from the global edge weights
            weights_subgraph = weights_g[:, subgraph_edge_indices]

            # The number of nodes in the current subgraph is just the length
            nnodes = vertex_idx.shape[0]


            # Next: Swap the first item with the current vertex number in the subgraph
            if vertex_idx[0] != vertex:
                # Find index of the current vertex and switch it with the first item
                switch_idx = (vertex_idx == vertex).nonzero().item()

                # Switch weights
                _weight_tmp = weights_subgraph[:, 0].clone().detach()
                weights_subgraph[:, 0] = weights_subgraph[:, switch_idx]
                weights_subgraph[:, switch_idx] = _weight_tmp

                # Switch vertex numbers
                vertex_idx[switch_idx] = vertex_idx[0]
                vertex_idx[0] = vertex

            # Build a new edge_index, where the current vertex, indexed by zero, is connected
            # to all other vertices. This is a directed graph

            # OBS! Ordering edge_index in this way [[0, 0, ..., 0], [0, 1, .., nnodes]]
            # requires to set flow="target_to_source" in MessagePassing classes
            edge_index_zero = torch.stack([torch.zeros(nnodes, dtype=torch.long),
                                        torch.arange(nnodes, dtype=torch.long)])

            graph_list.append(Data(x=data_x_g[vertex_idx, :], y=data_y_g[vertex, :],
                            edge_index=edge_index_zero.to(device=cuda),
                            edge_attr=weights_subgraph.T,
                            num_nodes=nnodes))


        torch.save(self.collate(graph_list), self.processed_paths[0])

        print("processed_paths = ", self.processed_paths)

        return None


class XGCGraphDataset_unnormalized(InMemoryDataset):
    r"""Loads output of xgc-kem as a list of graphs. Performs no normalization and does
    not remove zero-nodes.

    Args:
        root(string): Root directory where the bp files reside. The processed dataset will be saved in
            the same directory.
    """
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, tidx=4):
        self.tidx = tidx
        super(InMemoryDataset, self).__init__(root, transform, pre_transform, pre_filter)

        print("processed_paths = ", self.processed_paths)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        #print("raw_file_name called")
        #return ['some_file_1', 'some_file_2', ...]
        pass

    @property
    def processed_file_names(self):
        print("processed_file_names called")
        return [f"data_tidx_{self.tidx:05d}_unnorm.pt"]

    def _download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.

        torch.set_default_dtype(torch.float64)
        cuda = torch.device('cuda')

        print("Processing...")

        with adios2.open(join(self.root, "xgc.mesh.bp"), "r") as df:
            coords = df.read("/coordinates/values")
            nextnode = df.read("nextnode")
            nc = df.read("/cell_set[0]/node_connect_list")
            df.close()

        with adios2.open(join(self.root, "xgc.bfield.bp"), "r") as df:
            Bvec = df.read("/node_data[0]/values")
            #v2 = df.read("/node_data[1]/values")
            df.close()

        # Calculate the total magnetic field
        Btotal = np.sqrt(np.sum(Bvec**2.0, axis=1))  # Total magnetic field, in T
        # Get background density and temperature. Hard-code those for test cases.
        ne0 = 1e19 # Background electron density, in m^-3
        ni0 = 1e19 # Background ion density, in m^-3
        Te0 = 2e3 # Background electron temperaure, in eV
        Ti0 = 2e3 # Background ion temperature, in eV

        print(f"Using hard-coded value for ne0: {ne0}m^-3")
        print(f"Using hard-coded value for ni0: {ni0}m^-3")
        print(f"Using hard-coded value for Te0: {Te0}eV")
        print(f"Using hard-coded value for Ti0: {Ti0}eV")

        mu0 = 4. * np.pi * 1e-7 # Vacuum permeability, in H/m
        eps0 = 8.854e-12 # Vacuum permittivity, in F/m
        e = 1.602e-19 # Elementary charge, in C
        me = 9.109e-31 # Electron mass, in kg
        mi = 1.67e-27 # Proton mass, in kg
        c0 = 3e8 # Speed of light in m/s

        VA = Btotal / np.sqrt(mu0 * mi * ni0)
        VS = np.sqrt(e * Te0 / me)
        wpe = np.sqrt(ne0 * e * e / me / eps0)
        de = c0 / wpe

        print(f"mean(VA) = {VA.mean():4.2e}m/s")
        print(f"VS = {VS:4.2e}m/s")
        print(f"omega_pe = {wpe:4.2e}Hz")
        print(f"Skin depth de = {de:f}m")

        # Set parallel and perpendicular normalizations
        dt = 1e-8
        norm_par = 0.5 * (VA + VS) * dt
        norm_perp = de
        rbf = lambda x: 1./np.sqrt(1. + x*x)

        # Create the mesh and get the node connections
        mesh = xgc_grid.xgc_mesh(nc, coords)
        conns = xgc_grid.get_node_connection(nc, nextnode, local_plane=False)

        nodes_per_plane = max(conns.keys()) + 1
        num_planes = 8
        num_vertices = nodes_per_plane * num_planes
        print(f"Nodes per plane: {nodes_per_plane}, poloidal planes: {num_planes}")


        # Calculate connection weights based on cartesian distance an RBFs
        print(f"Calculating weights")
        all_weights_3 = []
        for from_node in conns.keys():
            dist_normed = xgc_grid.normalized_cartesian_distance(from_node, conns[from_node], coords, norm_par, norm_perp)
            all_weights_3.append(dist_normed)

        all_weights_3 = np.vstack(all_weights_3)


        print(f"Scaling weights")
        scaler_r = preprocessing.MinMaxScaler()
        all_weights_3 = scaler_r.fit_transform(all_weights_3)

        # Generate the weights tensor. Repeat the weights from the stencil of the first plane
        weights = torch.tensor(all_weights_3)
        weights = weights.repeat((num_planes, 1)).T
        print("Weights repeated over {num_planes} shape = ", weights.shape)


        print(f"Calculating connections in the mesh")
        all_conns = []
        for plane in range(num_planes):
            for from_node in conns.keys():
                rv = xgc_grid.shift_connections(from_node, conns[from_node], nodes_per_plane, plane, num_planes)
                all_conns += rv

        print(f"Calculating global edge index")
        edge_index_all = torch.tensor(all_conns).contiguous().T

        # Compile feature vectors on all nodes
        node_features = {}

        filename_x = f"xgc.3d.{self.tidx:05d}.bp"
        with adios2.open(join(self.root, filename_x), "r") as df:
            print(f"Loading vertex features from {filename_x}")
            for key in ["eden", "iden", "u_e", "u_i", "dpot", "a_par"]:
                node_features[key] = df.read(key, start=[0, 0], count=[nodes_per_plane, num_planes], step_start=0, step_count=1)

        for key in node_features.keys():
            node_features[key] = node_features[key][0, :, :].T.flatten()
        node_features["B"] = np.tile(Btotal, (num_planes, 1)).T.flatten()
        data_x = torch.tensor([v for v in node_features.values()]).T

        # Compile target features
        target_features = {}

        filename_y = f"xgc.3d.{(self.tidx+1):05d}.bp"
        with adios2.open(join(self.root, filename_y), "r") as df:
            for key in ["dpot", "a_par"]:
                target_features[key] = df.read(key, start=[0, 0], count=[nodes_per_plane, num_planes],
                                            step_start=0, step_count=1).T.flatten()

        data_y = torch.tensor([v for v in target_features.values()]).T


        # Generate graph list
        edge_index_g = edge_index_all.cuda()
        weights_g = weights.cuda()

        graph_list = build_graph_list(data_x, data_y, edge_index_g, weights_g)

        torch.save(self.collate(graph_list), self.processed_paths[0])

        print("processed_paths = ", self.processed_paths)

        return None


class XGC_it_dataset(InMemoryDataset):
    r"""Loads the graph-lists of ml_data_case_2 from converted numpy files.

    Args:
        root(string): Root directory where the bp files reside. The processed dataset will be saved in
            the same directory.
    """
    def __init__(self, root, idx_n=1, idx_k=1, transform=None, pre_transform=None, pre_filter=None):
        self.idx_n = idx_n
        self.idx_k = idx_k
        super(InMemoryDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        return [f"edge_index_all.pt",
                f"weights.pt",
                f"Btotal.pt",
                f"xgc.3d.{(self.idx_n - 1):05d}.c.npz",
                f"xgc.3d.{self.idx_n:05d}.c.npz",
                f"xgc.3d.{self.idx_n:03d}{self.idx_k:02d}.npz" ]


    @property
    def processed_file_names(self):
        return [f"xgc_mlcase2_graphlist_{self.idx_n:03d}{self.idx_k:02d}.pt"]


    def process(self):
        X_key_list = ["eden", "iden", "u_e", "u_i", "dpot", "a_par", "apar_res", "pot_res"]
        num_planes = 8
        fname_ei, fname_wts, fname_B, fname_prev, fname_conv, fname_iter = self.raw_paths

        edge_index_all = torch.load(fname_ei)
        weights = torch.load(fname_wts)
        Btotal = torch.load(fname_B)

        features_prev = {}
        with np.load(join(self.root, fname_prev)) as df:
            for key in ["eden", "iden", "u_e", "u_i", "dpot", "a_par"]:
                features_prev[key] = df[key].T.flatten()

        # Compile feature vectors
        features_node = {}
        with np.load(join(self.root, fname_iter)) as df:
            for key in X_key_list:
                features_node[key] = df[key].T.flatten()
            # Save apar_try and pot_try to calculate the error later
            apar_try = df["apar_try"].T.flatten()
            pot_try = df["pot_try"].T.flatten()

        features_node["B"] = np.tile(Btotal, (num_planes, 1)).T.flatten() / Btotal.max()

        eden_rel = features_node["eden"] / features_prev["eden"] - 1.0
        iden_rel = features_node["iden"] / features_prev["iden"] - 1.0
        ue_rel = features_node["u_e"] / features_prev["u_e"] - 1.0
        ui_rel = features_node["u_i"] / features_prev["u_i"] - 1.0
        apar_rel = features_node["a_par"] / features_prev["a_par"] - 1.0
        dpot_rel = features_node["dpot"] / features_prev["dpot"] - 1.0
        apar_res = features_node["apar_res"]
        dpot_res = features_node["pot_res"]
        data_x = torch.tensor([features_node["B"], eden_rel, iden_rel, ue_rel, ui_rel, apar_rel, dpot_rel, apar_res, dpot_res]).T

        # Compile target features
        features_target = {}
        with np.load(join(self.root, fname_conv)) as df:
            features_target["apar_err"] = (df["apar_try"].T.flatten() - apar_try)
            features_target["dpot_err"] = (df["pot_try"].T.flatten() - pot_try)
        data_y = torch.tensor([v for v in features_target.values()]).T

        data_list = build_graph_list(data_x, data_y, edge_index_all, weights, cuda=True)

        print(f"Build graph lists, len={len(data_list)}")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        print(data_list)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class XGC_it_dataset_kneighbor(InMemoryDataset):
    r"""Loads the graph-lists of ml_data_case_2 from converted numpy files.
    Normalizes ne, ni, ue, and ui to value at previous time step

    Args:
        root(string): Root directory where the bp files reside. The processed dataset will be saved in
            the same directory.
    """
    def __init__(self, root, idx_n=1, idx_k=1, depth=1, transform=None, pre_transform=None, pre_filter=None, map_location=None):
        self.idx_n = idx_n
        self.idx_k = idx_k
        self.depth = depth
        super(InMemoryDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], map_location=map_location)


    @property
    def raw_file_names(self):
        return [f"xgc.mesh.bp",
                f"xgc.bfield.bp",
                f"xgc.3d.{(self.idx_n - 1):05d}.c.npz",
                f"xgc.3d.{self.idx_n:05d}.c.npz",
                f"xgc.3d.{self.idx_n:03d}{self.idx_k:02d}.npz" ]


    @property
    def processed_file_names(self):
        print("Called processed files")
        return [f"xgc_mlcase2_depth{self.depth}_{self.idx_n:03d}{self.idx_k:02d}.pt"]


    def process(self):
        X_key_list = ["eden", "iden", "u_e", "u_i", "dpot", "a_par", "apar_res", "pot_res"]
        num_planes = 8
        fname_mesh, fname_B, fname_prev, fname_conv, fname_iter = self.raw_paths

        with adios2.open(join(self.root, "xgc.mesh.bp"), "r") as df:
            coords = df.read("/coordinates/values")
            nextnode = df.read("nextnode")
            nc = df.read("/cell_set[0]/node_connect_list")
            df.close()

        with adios2.open(join(self.root, "xgc.bfield.bp"), "r") as df:
            Bvec = df.read("/node_data[0]/values")
            df.close()

        # Calculate the total magnetic field
        Btotal = np.sqrt(np.sum(Bvec**2.0, axis=1))  # Total magnetic field, in T
        # Get background density and temperature. Hard-code those for test cases.
        ne0 = 1e19 # Background electron density, in m^-3
        ni0 = 1e19 # Background ion density, in m^-3
        Te0 = 2e3 # Background electron temperaure, in eV
        Ti0 = 2e3 # Background ion temperature, in eV

        mu0 = 4. * np.pi * 1e-7 # Vacuum permeability, in H/m
        eps0 = 8.854e-12 # Vacuum permittivity, in F/m
        e = 1.602e-19 # Elementary charge, in C
        me = 9.109e-31 # Electron mass, in kg
        mi = 1.67e-27 # Proton mass, in kg
        c0 = 3e8 # Speed of light in m/s

        VA = Btotal / np.sqrt(mu0 * mi * ni0)
        VS = np.sqrt(e * Te0 / me)
        wpe = np.sqrt(ne0 * e * e / me / eps0)
        de = c0 / wpe

        # Set parallel and perpendicular normalizations
        dt = 1e-8
        norm_par = 0.5 * (VA + VS) * dt
        norm_perp = de
        rbf = lambda x: 1./np.sqrt(1. + x*x)
        conns_3d = xgc_grid.get_node_connections_3d(nc, nextnode, 8)

        #######################################################################################
        # Load feature and target data from simulation data
        features_prev = {}
        with np.load(join(self.root, fname_prev)) as df:
            for key in ["eden", "iden", "u_e", "u_i", "dpot", "a_par"]:
                features_prev[key] = df[key].T.flatten()

        # Compile feature vectors
        features_node = {}
        with np.load(join(self.root, fname_iter)) as df:
            for key in X_key_list:
                features_node[key] = df[key].T.flatten()
            # Save apar_try and pot_try to calculate the error later
            apar_try = df["apar_try"].T.flatten()
            pot_try = df["pot_try"].T.flatten()

        features_node["B"] = np.tile(Btotal, (num_planes, 1)).T.flatten() / Btotal.max()

        eden_rel = features_node["eden"] / features_prev["eden"] - 1.0
        iden_rel = features_node["iden"] / features_prev["iden"] - 1.0
        ue_rel = features_node["u_e"] / features_prev["u_e"] - 1.0
        ui_rel = features_node["u_i"] / features_prev["u_i"] - 1.0
        apar_rel = features_node["a_par"] / features_prev["a_par"] - 1.0
        dpot_rel = features_node["dpot"] / features_prev["dpot"] - 1.0
        apar_res = features_node["apar_res"]
        dpot_res = features_node["pot_res"]
        data_x = torch.tensor([features_node["B"], eden_rel, iden_rel, ue_rel, ui_rel, apar_rel, dpot_rel, apar_res, dpot_res]).T

        # Compile target features
        features_target = {}
        with np.load(join(self.root, fname_conv)) as df:
            features_target["apar_err"] = (df["apar_try"].T.flatten() - apar_try)
            features_target["dpot_err"] = (df["pot_try"].T.flatten() - pot_try)
        data_y = torch.tensor([v for v in features_target.values()]).T

        #######################################################################################
        # Construct a graph from the global simulation domain
        num_planes =  8
        G = nx.Graph()

        all_k = list(conns_3d.keys())
        for k in all_k:
            for j in conns_3d[k]:
                G.add_edge(k, j)

        graph_list = []

        #######################################################################################
        # Construct a sub-graphs with specified depth from the global simulation domain
        #
        for root_vtx in list(G.nodes):
            subgraph = nx.Graph()
            sub_edges = nx.algorithms.traversal.breadth_first_search.bfs_edges(G, source=root_vtx, depth_limit=self.depth)
            for edge in sub_edges:
                wt = xgc_grid.normalized_cartesian_distance_3d(edge[0], [edge[1]], coords, norm_par, norm_perp)
                subgraph.add_edge(edge[0], edge[1], weight=rbf(wt))

            # Now convert the nx graph into a pytorch-geometric structure
            pyg_graph = utils.convert.from_networkx(subgraph)
            pyg_graph.x = data_x[list(subgraph.nodes), :]
            pyg_graph.y = data_y[list(subgraph.nodes), :]
            pyg_graph.weight = pyg_graph.weight.squeeze(1)
            pyg_graph.root_vtx = root_vtx
            graph_list.append(pyg_graph)

        if self.pre_filter is not None:
            graph_list = [data for data in graph_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            graph_list = [self.pre_transform(data) for data in graph_list]

        data, slices = self.collate(graph_list)
        print(f"Saving to {self.processed_paths[0]}")
        torch.save((data, slices), self.processed_paths[0])



class XGC_it_dataset_kneighbor_norm2(InMemoryDataset):
    r"""Loads the graph-lists of ml_data_case_2 from converted numpy files.
    Normalizes ne, ni, ue, and ui only by characteristic scale.

    Args:
        root(string): Root directory where the bp files reside. The processed dataset will be saved in
            the same directory.
    """
    def __init__(self, root, idx_n=1, idx_k=1, depth=1, transform=None, pre_transform=None, pre_filter=None, map_location=None):
        self.idx_n = idx_n
        self.idx_k = idx_k
        self.depth = depth
        super(InMemoryDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], map_location=map_location)


    @property
    def raw_file_names(self):
        return [f"xgc.mesh.bp",
                f"xgc.bfield.bp",
                f"xgc.3d.{self.idx_n:05d}.c.npz",
                f"xgc.3d.{self.idx_n:03d}{self.idx_k:02d}.npz" ]


    @property
    def processed_file_names(self):
        print("Called processed files")
        return [f"xgc_mlcase2_norm2_depth{self.depth}_{self.idx_n:03d}{self.idx_k:02d}.pt"]


    def process(self):
        X_key_list = ["eden", "iden", "u_e", "u_i", "dpot", "a_par", "apar_res", "pot_res"]
        num_planes = 8
        fname_mesh, fname_B, fname_conv, fname_iter = self.raw_paths

        with adios2.open(join(self.root, "xgc.mesh.bp"), "r") as df:
            coords = df.read("/coordinates/values")
            nextnode = df.read("nextnode")
            nc = df.read("/cell_set[0]/node_connect_list")
            df.close()

        with adios2.open(join(self.root, "xgc.bfield.bp"), "r") as df:
            Bvec = df.read("/node_data[0]/values")
            df.close()

        # Calculate the total magnetic field
        Btotal = np.sqrt(np.sum(Bvec**2.0, axis=1))  # Total magnetic field, in T
        # Get background density and temperature. Hard-code those for test cases.
        ne0 = 1e19 # Background electron density, in m^-3
        ni0 = 1e19 # Background ion density, in m^-3
        Te0 = 2e3 # Background electron temperaure, in eV
        Ti0 = 2e3 # Background ion temperature, in eV

        mu0 = 4. * np.pi * 1e-7 # Vacuum permeability, in H/m
        eps0 = 8.854e-12 # Vacuum permittivity, in F/m
        e = 1.602e-19 # Elementary charge, in C
        me = 9.109e-31 # Electron mass, in kg
        mi = 1.67e-27 # Proton mass, in kg
        c0 = 3e8 # Speed of light in m/s

        VA = Btotal / np.sqrt(mu0 * mi * ni0)
        VS = np.sqrt(e * Te0 / me)
        wpe = np.sqrt(ne0 * e * e / me / eps0)
        de = c0 / wpe

        # Set parallel and perpendicular normalizations
        dt = 1e-8
        norm_par = 0.5 * (VA + VS) * dt
        norm_perp = de
        rbf = lambda x: 1./np.sqrt(1. + x*x)
        conns_3d = xgc_grid.get_node_connections_3d(nc, nextnode, 8)

        #######################################################################################
        # Compile feature vectors
        features_node = {}
        with np.load(join(self.root, fname_iter)) as df:
            for key in X_key_list:
                features_node[key] = df[key].T.flatten()
            # Save apar_try and pot_try to calculate the error later
            apar_try = df["apar_try"].T.flatten()
            pot_try = df["pot_try"].T.flatten()

        features_node["B"] = np.tile(Btotal, (num_planes, 1)).T.flatten() / Btotal.max()

        # For statistics of ne, ni, ue, ui, apar and dpot fields over time, see explore_data_mldata_case2_normalizations.ipynb
        data_x = torch.tensor([features_node["B"],
                               features_node["eden"] * 1e-7, features_node["iden"] * 1e-7,
                               features_node["u_e"] * 1e-14, features_node["u_i"] * 1e-14,
                               features_node["a_par"] * 1e13, features_node["dpot"] * 1e7,
                               features_node["apar_res"], features_node["pot_res"]]).T

        # Compile target features
        features_target = {}
        with np.load(join(self.root, fname_conv)) as df:
            features_target["apar_err"] = (df["apar_try"].T.flatten() - apar_try)
            features_target["dpot_err"] = (df["pot_try"].T.flatten() - pot_try)
        data_y = torch.tensor([v for v in features_target.values()]).T

        #######################################################################################
        # Construct a graph from the global simulation domain
        num_planes =  8
        G = nx.Graph()

        all_k = list(conns_3d.keys())
        for k in all_k:
            for j in conns_3d[k]:
                G.add_edge(k, j)

        graph_list = []

        #######################################################################################
        # Construct a sub-graphs with specified depth from the global simulation domain
        #
        for root_vtx in list(G.nodes):
            subgraph = nx.Graph()
            sub_edges = nx.algorithms.traversal.breadth_first_search.bfs_edges(G, source=root_vtx, depth_limit=self.depth)
            for edge in sub_edges:
                wt = xgc_grid.normalized_cartesian_distance_3d(edge[0], [edge[1]], coords, norm_par, norm_perp)
                subgraph.add_edge(edge[0], edge[1], weight=rbf(wt))

            # Now convert the nx graph into a pytorch-geometric structure
            pyg_graph = utils.convert.from_networkx(subgraph)
            pyg_graph.x = data_x[list(subgraph.nodes), :]
            pyg_graph.y = data_y[list(subgraph.nodes), :]
            pyg_graph.weight = pyg_graph.weight.squeeze(1)
            pyg_graph.root_vtx = root_vtx
            graph_list.append(pyg_graph)

        if self.pre_filter is not None:
            graph_list = [data for data in graph_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            graph_list = [self.pre_transform(data) for data in graph_list]

        data, slices = self.collate(graph_list)
        print(f"Saving to {self.processed_paths[0]}")
        torch.save((data, slices), self.processed_paths[0])

# End of file pyg_xgc_loaders.py
