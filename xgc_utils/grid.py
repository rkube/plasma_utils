#!/usr/bin/python
#-*- Encoding: UTF-8 -*-


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
        self.vertices = (v1, v2, v3)
        self.coord_list = []
        
        for vertex in self.vertices:
            #print("Appending " + str(all_nodes[(all_nodes[:,0] == node).nonzero()][0,1:]))
            self.coord_list.append(list(coord_lookup[vertex, :]))
        
    def coords(self):
        """Returns a list of coordinates of the three vertices.
        """
        return self.coord_list
    
    
    def coords_R(self):
        """ Returns a list of the R-coordinates for the three vertices.
        """
        res = [c[0] for c in self.coord_list]
        return res 

    
    def coords_Z(self):
        """ Returns a list of the Z-coordinates for the three vertices.
        """
        res = [c[1] for c in self.coord_list]
        return res 


class xgc_mesh():
    """ Describes the triangle mesh for an XGC simulation.
    
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

    for node_num in range(conns.max()):
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
