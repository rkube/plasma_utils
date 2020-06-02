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



def get_node_connections_3d(conns: np.ndarray, nextnode: np.ndarray, num_planes : int = 8) -> dict:
    """For a given node, calculates the nodes it is directly connected to via triangles.
    Output is a dictionary with the reference node as the key and a list of directly connected vertices as the value.

    Input:
    ======
    conns: Array of vertices in a triangle. shape = (num_triangles, 3),
    nextnode: Array that lists the vertex at the next plane.

    Output:
    =======
    conn_dict: Dictionary. Key: vertex index. Value: List of vertex indices that a node connects to.

    Here is some mockup-code that is automated by this routine.

    # Total number of planes
    n_pl = 4
    # Vertices per plane
    n_per_pl = 8

    Create an array of vertices a reference vertex is connected to.
    The last two refer to entries from nextnode.
    i_0 = np.array([1, 2, 3, 2 + n_per_pl, 1 - n_per_pl])
    # Plane 1
    i_1 = np.array([1, 2, 3, 2 + n_per_pl, 1 - n_per_pl]) + 1 * n_per_pl
    # Plane 2
    i_2 = np.array([1, 2, 3, 2 + n_per_pl, 1 - n_per_pl]) + 2 * n_per_pl
    # Plane 3
    i_3 = np.array([1, 2, 3, 2 + n_per_pl, 1 - n_per_pl]) + 3 * n_per_pl

    print(f"Plane 0: {i_0}")
    print(f"Plane 1: {i_1}")
    print(f"Plane 2: {i_2}")
    print(f"Plane 3: {i_3}")

    Plane 0: [ 1  2  3 10 -7]
    Plane 1: [ 9 10 11 18  1]
    Plane 2: [17 18 19 26  9]
    Plane 3: [25 26 27 34 17]

    We see that the array, describing connections in plane 0, refers to -7. A node
    in plane 3 (due to periodicity). The array  translated into plane 3, refers to
    node 34. We can map these indices onto the range of vertices, [0:31], by using mod:

    print(f"Plane 0, with mod: {np.mod(i_0, n_per_pl * n_pl)}")
    print(f"Plane 3, with mod: {np.mod(i_3, n_per_pl * n_pl)}")

    Plane 0, with mod: [ 1  2  3 10 25]
    Plane 3, with mod: [25 26 27  2 17]

    """

    conn_dict = {}

    # This + 1 is important!
    vtx_per_plane = conns.max() + 1
    assert(conns.min() == 0)

    # Iterate over all vertices.
    for vertex in range(vtx_per_plane):
        # 1) Find all triangles which include the current vertex
        # Since each vertex occurs either 0 or 1 time in a triangle, each row
        # returned from np.argwhere is unique.
        idx_tri = np.argwhere(conns == vertex)[:, 0]

        # Now we have the triangles in which vertex occurs, identified as rows in conns.
        # Get the unique vertices for these triangles
        connected_vertices = np.unique(conns[idx_tri, :])

        # Now we need to find the cross-plane connections.
        # Find the first cross-plane connection via a direct look-up from nextnode
        connected_next_plane = nextnode[vertex] + vtx_per_plane
        # Find the previous one via a inverse look-up
        try:
            connected_prev_plane = np.argwhere(nextnode == vertex).item() - vtx_per_plane
        except:
            # This may actually fail. In this case we just assume a self-connection
            connected_prev_plane = vertex - vtx_per_plane

        conn_vtx_list = np.concatenate((connected_vertices, np.array([connected_next_plane, connected_prev_plane])))

        # Now we insert conn_vtx_list into conn_dict as the value for the current vertex.
        # Do this for all planes and apply modulo for the first and last plane

        for idx_pl in range(num_planes):
            # We are inserting a vertex shifted to the plane at the current iteration
            vtx_shift = vertex + idx_pl * vtx_per_plane
            if((idx_pl == 0) or (idx_pl == num_planes - 1)):
                conn_dict.update({vtx_shift: np.mod(conn_vtx_list + idx_pl * vtx_per_plane, num_planes * vtx_per_plane)})
            else:
                conn_dict.update({vtx_shift: conn_vtx_list + idx_pl * vtx_per_plane})

    return conn_dict


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


def normalized_cartesian_distance(root_vtx: int, conns: list, coords: np.ndarray, norm_par: np.ndarray, norm_perp: np.ndarray, num_planes: int=8) -> np.ndarray:
    """Calculates the normalized cartesian distance from node root_vtx to each item in conns.
    The value to normalize is taken to be at the to-node. I.e. we assume that there is little variation
    in value we normalize to.

    Input:
    ======
    root_vtx: int, from node
    conns: list, to nodes.
    coords: array, shape(nnodes, 2). Vector of R,Z coordinates.
    norm_par: array, shape(nnodes)
    norm_perp: array, shape(nnodes)
    num_planes: int, number of toroidal planes

    Returns:
    ========
    distances: list of cartesian distances.
    """

    print("USING THE NEW LAYOUT FOR conns, see get_node_connections_3d")

    R0, Z0 = coords[root_vtx]
    # Create array from conection list
    conns_vec = np.array(conns)
    R_vec = coords[conns_vec, 0]
    Z_vec = coords[conns_vec, 1]

    # Get perpendicular normalization
    # If we have passed a ndarray as the perpendicular normalization, we have to use only the
    # appropriate items.
    norm_perp = norm_perp[conns_vec]

    # Multiply S_vec elementwise by 0 if c[1] == 0, by 1 if abs(c[1]) == 1
    S_mask = np.abs(conns_vec[:, 1])
    # Calculate all S values. Use the R value of the to-nodes
    S_vec = 2. * np.pi * R_vec / num_planes / 4.
    # Get the parallel normalization
    norm_par = norm_par[conns_vec]

    S_vec = S_vec * S_mask / norm_par

    dist_R = (R_vec - R0) / norm_perp
    dist_Z = (Z_vec - Z0) / norm_perp
    dist_S = S_vec

    dist = np.stack((dist_R, dist_Z, dist_S), axis=1)

    return(dist)


def normalized_cartesian_distance_old(nodeidx0, conns, coords, norm_par, norm_perp, num_planes=8):
     """Calculates the normalized cartesian distance from node nodeix0 to each item in conns.
     The value to normalize is taken to be at the to-node. I.e. we assume that there is little variation
     in value we normalize to.

     This is the old version of this code. It assumes allows for scalar as well as vector
     norm_perp and norm_par.

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

def normalized_cartesian_distance_3d(root_vtx: int, conns: list, coords: np.ndarray, norm_par: np.ndarray, norm_perp: float, num_planes: int = 8) -> np.ndarray:
    """Calculates the normalized cartesian distance from node root_vtx to each item in conns.
    The value to normalize is taken to be at the to-node. I.e. we assume that there is little variation
    in value we normalize to.

    Use this in combination with get_node_connections_3d

    Input:
    ======
    root_vtx: int, from vertex
    conns: list, to vertices
    coords: array, shape(nnodes, 2). Vector of R,Z coordinates.
    norm_par: shape(nnodes), Parallel normalization (depends on B!)
    norm_perp: Perpendicular normalization constant
    num_planes: int, number of toroidal planes

    Returns:
    ========
    distances: list of cartesian distances.
    """

    num_vtx = coords.shape[0]

    R0, Z0 = coords[np.mod(root_vtx, num_vtx)]
    # Create array from conection list
    conns_vec = np.array(conns)
    R_vec = coords[np.mod(conns_vec, num_vtx), 0]
    Z_vec = coords[np.mod(conns_vec, num_vtx), 1]

    # Find the z-planes on which each vertex lies
    zplane_root = root_vtx // num_vtx
    zplane_conns = conns_vec // num_vtx

    zplane_delta = np.abs(zplane_conns - zplane_root)

    # Calculate all S values. Use the R value of the to-nodes
    S_vec = 2. * np.pi * R_vec * zplane_delta / num_planes / 4.
    # Get the parallel normalization
    norm_par = norm_par[np.mod(conns_vec, num_vtx)]

    dist_R = (R_vec - R0) / norm_perp
    dist_Z = (Z_vec - Z0) / norm_perp
    dist_S = S_vec / norm_par

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
