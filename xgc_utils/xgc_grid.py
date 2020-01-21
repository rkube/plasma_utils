#!/usr/bin/python
#-*- Encoding: UTF-8 -*-

import string
import numpy as np

class mesh_triangle_2d():
    """Describes triangles in the grid.
    Members:
    coord_list:  List [(R_i, phi_i), i = 1,2,3] for the three vertices
    node_list:  
    (v1, v2 v3):   Tuple of vertices that the triangle consitutes
    """
    
    def __init__(self, v1, v2, v3, coord_lookup):
        """Describes the coordinates of a triangle in the mesh
        Input:
        ======
        v1, v2, v3:   The vertex numbers constituting the triangle
        coord_lookup:   Array with the coordinates of the vertices. This is "/coordinates/values" from xgc.mesh.bp
        """
        self.vertices = (int(v1), int(v2), int(v3))
        self.coord_list = []
        
        for vertex in self.vertices:
            #print("Appending " + str(all_nodes[(all_nodes[:,0] == node).nonzero()][0,1:]))
            self.coord_list.append(list(coord_lookup[vertex, :]))
        
    def coords(self):
        """Returns a list of coordinates of the three vertices."""
        return self.coord_list
    
    
    def coords_R(self):
        """Returns a list of the R-coordinates for the three vertices."""
        res = [c[0] for c in self.coord_list]
        return res 

    
    def coords_Z(self):
        """Returns a list of the Z-coordinates for the three vertices."""
        res = [c[1] for c in self.coord_list]
        return res 

    def __str__(self):
        """Fancy printing."""
        return "({0:6.4f}, {1:6.4f}, {2:6.4f})".format(*self.vertices)


class xgc_mesh():
    """ Describes the triangle mesh for an XGC simulation.

    A typical workflow when working with XGC is to generate the mesh class
    from coords and node_connect_list variables stored in xgc.mesh.bp like this


    >>> import adios2
    >>> with adios2.open(join(datadir, "xgc.mesh.bp"), "r") as df:
            coords = df.read("/coordinates/values")
            nc = df.read("/cell_set[0]/node_connect_list")
    >>> mesh = xgc_grid.xgc_mesh(nc, coords)

    
    """
    def __init__(self, conns, coords):
        """
        Input:
        ======
        conns: Connections of the vertices. An array of [num_triangles, 3]. This is 'nd_connect_list' from xgc.mesh.bp.
        coords: Coordinates of the vertices. An array of [R, Z]. This is '/coordinates/values' from xgc.mesh.bp


        """
        self.triangle_list = []
        self.num_triangles = conns.shape[0]
        self.num_vertices = coords.shape[0]
        
        for tri in range(self.num_triangles):
            self.triangle_list.append(mesh_triangle_2d(conns[tri, 0], 
                                                       conns[tri, 1],
                                                       conns[tri, 2], coords))  
        
    def get_triangles(self):
        """ Returns the list of triangles that constitute the simulation domain.
        """
        
        return self.triangle_list


    def __str__(self):
        pstr = "A mesh consisting of"
        pstr += f" {self.num_triangles:8d} triangles"
        pstr += f" {self.num_vertices:8d} vertices"

        return(pstr)




def get_node_connection(conns, nextnode, local_plane=True, pidx=0):
    """Generates a list of nodes to which a given node is connected.
    Output is a dictionary where the node number is the key and the list of nodes is the value.

    Input:
    ======
    conns: Connections of the vertices. This is 'nd_connect_list' from xgc.mesh.bp
    nextnode: Approximate next node in toroidal direction. This is 'nextnode' from xgc.mesh.bp
    local_plane: If True, do not include connections across the plane. If false, add connections across toroidal planes.
    pidx: index of the toroidal plane.

    Returns:
    ========
    conn_dict: Dictionary of type {node number: [list of connected nodes]}
    """

    import numpy as np

    conn_dict = {}

    for node_num in range(conns.max() + 1):
        # Index, where the current node appears in cons
        idx_node = np.argwhere(conns == node_num)[:,0]
        # List of 
        connections_in_plane = conns[idx_node].ravel()
        conn_list = [(c, pidx) for c in np.unique(connections_in_plane)]

        if local_plane==False:
            #print("here")
            # Look for connections across the plane. nextnode is 
            # Each node always has a node it connects to in the next plane
            conn_list.append((nextnode[node_num], pidx + 1))
            try:
                conn_list.append((np.argwhere(nextnode == node_num).item(), pidx - 1))
            except:
                pass

        # Add connection to next plane and from next plane(inverse look-up through np.argwhere)
        #res = np.hstack([connections_in_plane, np.array(nextnode[node_num]), np.argwhere(nextnode == node_num).ravel()])
        # Remove duplicate elements
        conn_dict[node_num] = conn_list

    return conn_dict


# End of file grid.py


def cartesian_distance(nodeidx0, conns, coords, num_planes=8):
    """Calculates the cartesian distance from node nodeidx0 to each item in conns.
    Input:
    ======
    nodeidx0: int, from-node
    conns: list of tuples, List of to-nodes in the form (to-node, +-1) where the +-1 denotes 
           a node in the next/previous toroidal plane
    coords: array, shape(nnodes, 2). Vector of R,Z coordinates. Dim0: nodes, dim1: R/Z
    num_planes: int, number of toroidal planes
    
    Returns:
    ========
    distances: list of cartesian distances.
    """
    
    R0, Z0 = coords[nodeidx0]
    # Create array from conection list
    conns_vec = np.array(conns)
    R_vec = coords[conns_vec[:, 0], 0]
    Z_vec = coords[conns_vec[:, 0], 1]

    # Calculate all S values. Use the R value of the to-nodes
    S_vec = 2. * np.pi * R_vec / num_planes / 4.
    # Multiply S_vec elementwise by 0 if c[1] == 0, by 1 if abs(c[1]) == 1
    S_mask = np.abs(conns_vec[:, 1])
    S_vec = S_vec * S_mask

    # Calculate cartesian distance
    dist = np.sqrt((R_vec - R0)**2. + (Z_vec - Z0)**2. + S_vec**2.)

    return(dist)


def normalized_cartesian_distance(nodeidx0, conns, coords, norm_par, norm_perp, num_planes=8):
    """Calculates the normalized cartesian distance from node nodeix0 to each item in conns. 
    The value to normalize is taken to be at the to-node. I.e. we assume that there is little variation
    in value we normalize to.
    
    Input:
    ======
    nodeidx0: int, from node
    conns: list, to nodes
    coords: array, shape(nnodes, 2). Vector of R,Z coordinates.
    norm_par: array, shape(nnodes)
    norm_perp: array, shape(nnodes)
    num_planes: int, number of toroidal planes
    
    Returns:
    ========
    distances: list of cartesian distances.
    """
    
        
    R0, Z0 = coords[nodeidx0]
    # Create array from conection list
    conns_vec = np.array(conns)
    R_vec = coords[conns_vec[:, 0], 0]
    Z_vec = coords[conns_vec[:, 0], 1]
    
    # Get perpendicular normalization
    # If we have passed a ndarray as the perpendicular normalization, we have to use only the
    # appropriate items.
    if(isinstance(norm_perp, np.ndarray)):
        norm_perp = norm_perp[conns_vec[:,0]]
    #elif(isinstance(norm_perp), float):
    #    norm_perp_vec = norm_perp * np.ones_like(R_vec)
    
    # Multiply S_vec elementwise by 0 if c[1] == 0, by 1 if abs(c[1]) == 1
    S_mask = np.abs(conns_vec[:, 1])
    # Calculate all S values. Use the R value of the to-nodes
    S_vec = 2. * np.pi * R_vec / num_planes / 4.
    # Get the parallel normalization
    if(isinstance(norm_par, np.ndarray)):
        norm_par = norm_par[conns_vec[:, 0]] 

    S_vec = S_vec * S_mask / norm_par
    
    
    dist_R = (R_vec - R0) / norm_perp
    dist_Z = (Z_vec - Z0) / norm_perp
    dist_S = S_vec 

    dist = np.stack((dist_R, dist_Z, dist_S), axis=1)


    return(dist)

def shift_connections(from_node, conns, n_per_plane, to_plane, num_planes):
    """Translate a list of connections to another plane.
    
    Inputs:
    =======
    from_node, int: The from-node.
    conns, list of tuples. List of tuples the from-node connects to on plane 0. Format (to_node, shift_plane).
                           shift_plane = +-1 denotes a connection to the next/previous plane.
    n_per_plane, int: Nodes per plane
    to_plane, int: To which plane we shift.
    num_planes, int: Number of total planes.
    
    Returns:
    ========
    conns_shifted, list of tuples. Lists the nodes from_node connects to. Format (from_node, to_node).
    """
    
    res = [(from_node + (to_plane * n_per_plane), 
            c[0] + ((to_plane + c[1]) % num_planes) * n_per_plane) for c in conns]
    return(res)
