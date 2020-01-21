# Encoding: UTF-8 -*-

import numpy as np
import torch

def get_zero_index(input_features, eps=1e-16):
    """Generates a boolean array that marks elements whose value is below a threshold with zero.

    Input is a dict of input features, each one an array and all with the same shape. We want to calculate of any 
    input feature on a given note is below a threshold. The returned array is the same shape as all the
    input features. It is true, if any input feature is below eps.

    Input:
    ======
    input_features: dict of input features. Each value is assumed to by of shape [n_per_plane * poloidal_planes]
    eps, float. The threshold we test each value against

    Returns:
    ========
    nzero_idx_bool: Boolean array of shape [n_per_plane * poloidal_plane].
    """

    # Find feature vectors where any feature is zero
    # _zero_idx: dim0: index of the node-features [n_per_plane * poloidal_planes], dim1: feature name
    _zero_idx = np.zeros([list(input_features.values())[0].shape[0], len(list(input_features.keys()))])

    for idx, values in enumerate(input_features.values()):
        _zero_idx[:, idx] = abs(values) < eps
    zero_idx = np.argwhere(_zero_idx.any(axis=1)).ravel()
    _zero_idx = 0

    # Create a boolean index array which indices only positive values in the fields
    nzero_idx_bool = np.ones(list(input_features.values())[0].shape[0], dtype=np.bool)
    nzero_idx_bool[zero_idx] = False

    print("Zero_idx: {0:d}/{1:d} elements below threshold".format(zero_idx.shape[0], 
                                                                  list(input_features.values())[0].shape[0]))

    return(nzero_idx_bool)


def amputate_zero_connections(nzero_idx_bool, edge_index):
    """Amputates connections to zero-feature-vectors in a graph.

    Input:
    ======
    nzero_idx_bool: ndarray, bool.
    edge_index: Graph connectivity in COO shape, see https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html


    Returns:
    ========
    zi_next: tensor, bool. Index-tensor that can be used to generate zero-amputated edge_index and weights.


    Example usage:
    data_x and data_y are scaled input and target feature vectors
    edge_index and weights describe node connections and weights.


    >>> nzero_idx_bool = get_zero_index(input_features)
    >>> zi_next = amputate_zero_connections(nzero_idx_bool edge_index)
    Now we create a graph Data structure using ~zi_next to 'amputate' zero connections.
    >>> data = Data(x=data_x, edge_index=edge_index[:, ~zi_next], edge_attr=weights[~zi_next], y=data_y)

    """

    cuda = torch.device('cuda')

    # Recover the indices where a zero-element is places in the feature-vector
    zero_idx = torch.where(torch.tensor(nzero_idx_bool))[0]
    
    # Move both, zero_idx and edge_index to the GPU
    zero_idx_g = zero_idx.to(device=cuda)
    edge_index_g = edge_index.to(device=cuda)

    #%%timeit -n1
    # 3.6 s ± 100 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    # Create boolean index array where we mark connections to a zero-node with True
    zi_next_g = torch.zeros(edge_index.shape[1], dtype=torch.bool, device=cuda)

    for zi in zero_idx_g:
        # Mark locations of current zero-node in edge_index_g as True
        zi_current_g = (edge_index_g == zi).any(0)
        # Update locations in zi_next_g
        zi_next_g = zi_next_g | zi_current_g
        
    print("Indexing {0:d} vertices to zero".format(zi_next_g.sum()))

    zi_next = zi_next_g.cpu()
    return(zi_next)


# End of file gcn_helpers.py