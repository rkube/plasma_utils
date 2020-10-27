# -*- Encoding: UTF-8 -*-

"""
Defines helper classes that provide squashed subgraphs for an xgc grid
"""

from os.path import join

import numpy as np
import adios2
import torch
import json

from ..xgc_utils.xgc_grid import get_node_connections_3d, normalized_cartesian_distance_3d
from .subgraphs import kneighbor_squashed


mu0 = 4. * np.pi * 1e-7 # Vacuum permeability, in H/m
eps0 = 8.854e-12 # Vacuum permittivity, in F/m
e = 1.602e-19 # Elementary charge, in C
me = 9.109e-31 # Electron mass, in kg
mi = 1.67e-27 # Proton mass, in kg
c0 = 3e8 # Speed of light in m/s

class xgc_grid_squashed():
    """
    Defines a squashed xgc-grid
    """

    def __init__(self, data_dir, ne0=1e10, ni0=1e19, Te0=2e3, Ti0=2e3, depth=3):
        """
        Parameters:
        -----------
        data_dir..: root directory of the XGC simulation
        ne0, ni0..: Background electron/ion density, in m^-3
        Te0, Ti0..: Backgroun electron/ion temperatures, in eV
        depth.....: k-neighborhood depth
        """
        self.data_dir = data_dir
        self.ne0 = ne0 # Background electron density, in m^-3
        self.ni0 = ni0 # Background ion density, in m^-3
        self.Te0 = Te0 # Background electron temperaure, in eV
        self.Ti0 = Ti0 # Background ion temperature, in eV
        self.depth = depth

        with adios2.open(join(data_dir, "xgc.mesh.bp"), "r") as df:
            self.coords = df.read("/coordinates/values")
            nextnode = df.read("nextnode")
            nc = df.read("/cell_set[0]/node_connect_list")

        with adios2.open(join(data_dir, "xgc.bfield.bp"), "r") as df:
            Bvec = df.read("/node_data[0]/values")

        # Calculate the total magnetic field and normalization fields
        Btotal = np.sqrt(np.sum(Bvec**2.0, axis=1))  # Total magnetic field, in T
        VA = Btotal / np.sqrt(mu0 * mi * ni0)
        VS = np.sqrt(e * self.Te0 / me)
        wpe = np.sqrt(self.ne0 * e * e / me / eps0)
        de = c0 / wpe
        # Set parallel and perpendicular normalizations
        dt = 1e-8
        self.norm_par = 0.5 * (VA + VS) * dt
        self.norm_perp = de
        #rbf = lambda x: 1./np.sqrt(1. + x*x)

        self.json_fname = join(data_dir, f"xgc_subgraphs_depth{self.depth:1d}.json")

        # Try loading the neighborhood graphs from a json file.
        # If that doesn't work, generate the graphs from the grid.
        try:
            with open(self.json_fname) as df_graph:
                tmp = json.load(df_graph)
                # By default, keys in json are strings. Convert them to int64.
                # Also convert the lists to numpy arrays
                self.k_neighbors = {np.int64(k): np.array(v, dtype=np.int64) for k, v in tmp.items()}
            print("Loaded k-neighbor subgraphs from file")

        except:
            print("Generating k-neighbor subgraphs from scratch")
            conns_3d = get_node_connections_3d(nc, nextnode, 8)
            all_vertices = list(conns_3d.keys())
            # Generate k-neighborhood for a vertex
            k_neighbors_tmp = {}
            for root_vtx in all_vertices:
                neighbors = set()
                neighbors = kneighbor_squashed(root_vtx, conns_3d, neighbors, self.depth)

                k_neighbors_tmp[int(root_vtx)] = list(neighbors)

            def convert(o):
                if isinstance(o, np.int64):
                    return int(o)
                raise TypeError
            with open(self.json_fname, "w") as df_graph:
                json.dump(k_neighbors_tmp, df_graph, default=convert)

            # Convert values to int64 arrays
            self.k_neighbors = {np.int64(k): np.array(v, dtype=np.int64) for k, v in k_neighbors_tmp.items()}


    def all_vertices(self):
        """Used for iteration over vertices"""

        return self.k_neighbors.keys()


    def gen_edge_index(self, root_vtx):
        """Generates a pytorch-geometric edge-index structure that describes
        the neighborhood-graph around root_vtx.

        Parameters:
        -----------
        root_vtx..: int, root-vertex for neighborhood
        """

        #
        # Use numpy routines for calculations and instantiate a torch.tensor
        # when returning. numpy routines are faster than torch routines for
        # the workload below
        #
        # Create a 2-tensor with the vertex-numbers as entries
        neighbors = self.k_neighbors[root_vtx]
        num_neighbors = len(self.k_neighbors[root_vtx])
        edge_index_tmp = np.ones([2, num_neighbors], dtype=np.int64)
        edge_index_tmp[0, :] = root_vtx
        edge_index_tmp[1, :] = np.array(list(neighbors))
        # Next transform the edge_index structure to zero-based, consecutive indices.
        # Do this by call enumerate the unique elements in edge_index_tmp
        # and replace all occurance of a given value in the old edge_index
        # with its index in the new array wherever this value occurs
        edge_index = np.zeros_like(edge_index_tmp)
        for idx, val in enumerate(np.unique(edge_index_tmp)):
            edge_index[edge_index_tmp == val] = idx

        return torch.tensor(edge_index)


    def weights(self, root_vtx):
        """Calculate normalized cartesian distance from a root-vertex
        to vertices in its k-neighborhood

        Parameters:
        -----------
        root_vtx..: int, root-vertex

        """

        neighbors = list(self.k_neighbors[root_vtx])
        weights = normalized_cartesian_distance_3d(root_vtx,
                                                   neighbors,
                                                   self.coords,
                                                   self.norm_par,
                                                   self.norm_perp,
                                                   num_planes=8)

        return weights


    def root_vertex_idx(self, edge_index):
        """Given an edge_index, find the index of the root vertex"""

        return(torch.nonzero(edge_index[1, :] == edge_index[0, 0]).item())






# End of file grid_graphs.py
