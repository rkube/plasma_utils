# -*- Encoding: UTF-8 -*-
from os.path import join

# Add the file's parent directory to the path so that we can import xgc_utils
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import numpy as np
import torch
import timeit
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric import utils

from sklearn import preprocessing
from itertools import product

sys.path.append("/home/rkube/software/adios2-devel/lib/python3.8/site-packages")
import adios2
from xgc_utils import xgc_grid, xgc_helpers
from .subgraphs import kneighbor_squashed

################################################################################
#                                                                              #
#        Loaders for models that include data from multiple iterations         #
#                                                                              #
################################################################################

class XGC_squashed_iter(InMemoryDataset):
    r"""Uses squashed graph neighborhoods where the root vertex connects directly
    to all vertices at max k edges away.

    Normalizes ne, ni, ue, and ui only by characteristic scale.

    Args:
        root(string): Root directory where the bp files reside. The processed dataset will be saved in
            the same directory.
    """
    def __init__(self, root, idx_n=1, depth=1, idx_k_list=[1, 2, 3], transform=None, pre_transform=None, pre_filter=None):
        self.idx_n = idx_n
        self.idx_k_list = idx_k_list
        self.depth = depth
        super(InMemoryDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        list_1 = [f"xgc.mesh.bp",
                  f"xgc.bfield.bp",
                  f"xgc.3d.{self.idx_n:05d}.c.npz"]

        list_2 = [f"xgc.3d.{self.idx_n:03d}{idx_k:02d}.npz" for idx_k in self.idx_k_list]

        return list_1 + list_2

    @property
    def processed_file_names(self):
        print("Called processed files")
        return [f"xgc_squashed_iter{self.idx_k_list[-1]:02d}_depth{self.depth:1d}_{self.idx_n:03d}.pt"]

    def process(self):
        print("Called process")
        X_key_list = ["eden", "iden", "u_e", "u_i", "dpot", "a_par", "apar_res", "pot_res"]
        num_planes = 8

        fname_mesh, fname_B, fname_conv, *fname_iter_list = self.raw_paths
        with adios2.open(fname_mesh, "r") as df:
            coords = df.read("/coordinates/values")
            nextnode = df.read("nextnode")
            nc = df.read("/cell_set[0]/node_connect_list")
            df.close()

        with adios2.open(fname_B, "r") as df:
            Bvec = df.read("/node_data[0]/values")
            df.close()

        # Calculate the total magnetic field
        Btotal = np.sqrt(np.sum(Bvec**2.0, axis=1))  # Total magnetic field, in T
        # Get background density and temperature. Hard-code those for test cases.
        ne0, ni0 = 1e19, 1e19 # Background electron density, in m^-3
        Te0, Ti0 = 2e3, 2e3 # Background electron temperaure, in eV

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
        conns_3d = xgc_grid.get_node_connections_3d(nc, nextnode, 8)
        all_vertices = list(conns_3d.keys())


        ########################################################################
        # Data_x_list is a list of torch.tensors, each tensor corresposnds to
        # an iteration index idx_k
        data_x_list = []
        for idx_k, fname_iter in zip(self.idx_k_list, fname_iter_list):
            print(f"Collecting data for iteration {idx_k:02d}")

            tic = timeit.default_timer()
            features_node = {}
            with np.load(join(self.root, fname_iter)) as df:
                for key in X_key_list:
                    features_node[key] = df[key].T.flatten()
                # Save apar_try and pot_try to calculate the error later
                apar_try = df["apar_try"].T.flatten()
                pot_try = df["pot_try"].T.flatten()

            features_node["B"] = np.tile(Btotal, (num_planes, 1)).T.flatten() / Btotal.max()
            features_node["iter"] = np.ones_like(features_node["B"]) * idx_k

            # For statistics of ne, ni, ue, ui, apar and dpot fields over time, see explore_data_mldata_case2_normalizations.ipynb
            data_x = torch.tensor([features_node["B"],
                                   features_node["iter"],
                                   features_node["eden"] * 1e-7, features_node["iden"] * 1e-7,
                                   features_node["u_e"] * 1e-14, features_node["u_i"] * 1e-14,
                                   features_node["a_par"] * 1e13, features_node["dpot"] * 1e7,
                                   features_node["apar_res"], features_node["pot_res"]]).T

            data_x_list.append(data_x)

            if idx_k == self.idx_k_list[-1]:

                # Compile target features
                features_target = {}
                with np.load(join(self.root, fname_conv)) as df:
                    features_target["apar_err"] = (df["apar_try"].T.flatten() - apar_try)
                    features_target["dpot_err"] = (df["pot_try"].T.flatten() - pot_try)
                data_y = torch.tensor([v for v in features_target.values()]).T

            toc = timeit.default_timer()
            print(f"Compiling features took {(toc-tic):6.4f}s")

        #########################################################################
        # Construct squashed sub-graphs with specified depth
        #
        # List of all graphs
        graph_list = []

        tic = timeit.default_timer()
        # Iterate over vertices and build the squashed neighborhood
        for root_vtx in all_vertices:
            neighbors = set()
            neighbors = kneighbor_squashed(root_vtx, conns_3d, neighbors, self.depth)

            neighbors = list(neighbors)
            num_neighbors = len(neighbors)

            edge_index_tmp = torch.ones([2, num_neighbors], dtype=torch.long)
            edge_index_tmp[0, :] = root_vtx
            edge_index_tmp[1, :] = torch.LongTensor(neighbors)

            # We also need to transform the edge_index structure to zero-based, consecutive
            # indices.
            # Do this by call enumerate the unique elements in edge_index_tmp
            # and replace all occurance of a given value in the old edge_index
            # with its index in the new array wherever this value occurs
            edge_index = torch.zeros_like(edge_index_tmp)
            for idx, val in enumerate(torch.unique(edge_index_tmp)):
                edge_index[edge_index_tmp == val] = idx

            weights = xgc_grid.normalized_cartesian_distance_3d(root_vtx, neighbors, coords, norm_par, norm_perp)

            # We have converted the grid-structure to graphs. That is, for a given
            # root vertex we have the indices of the neighboring vertices.
            # Now we need to generate the value-vectors at these vertices for all
            # iterations and build the Data object with them.
            # We want to pass these vectors as keys with name x1, x2, etc.
            # To do this we need to pass a dictionary that is consumed by **kwargs in the __init__ call:
            #
            # https://stackoverflow.com/questions/18056132/dynamically-add-keyword-arguments
            x_dict = {}
            for idx_k, data_x in zip(self.idx_k_list, data_x_list):
                key_name = f"x_{idx_k:02d}"
                x_dict[key_name] = data_x[neighbors, :]

            graph_list.append(Data(edge_index=edge_index, edge_attr=torch.DoubleTensor(weights),
                                   x=None, y=data_y[root_vtx, :], root_vtx=root_vtx,
                                   **x_dict))

        toc = timeit.default_timer()
        num_graphs = len(graph_list)

        print(f"Compiling {num_graphs:05d} graphs took {(toc-tic):6.3f}s")

        tic = timeit.default_timer()
        if self.pre_filter is not None:
            graph_list = [graph for graph in graph_list if self.pre_filter(graph)]
        toc = timeit.default_timer()
        num_graphs_filtered = len(graph_list)
        print(f"Applying pre-filter took {(toc - tic):6.4f}s - {num_graphs_filtered} graphs in list")

        tic = timeit.default_timer()
        if self.pre_transform is not None:
            graph_list = [self.pre_transform(data) for data in graph_list]
        toc = timeit.default_timer()
        print(f"Applying pre-transform took {(toc - tic):6.4f}s")

        data, slices = self.collate(graph_list)
        print(f"Saving to {self.processed_paths[0]}")
        torch.save((data, slices), self.processed_paths[0])





# End of file pyg_xgc_loaders2.py
