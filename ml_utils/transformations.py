# -*- Encoding: UTF-8 -*-

import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data

from .subgraphs import build_subgraph

def nan_filter(data):
    """Returns
    True, if no element of array is nan
    False if any element of array is nan."""
    return ~torch.any(torch.isnan(data.x))


def sqrt3_trf(X, subtract_med=True):
    """Transforms data as Y = sgn(X) |X|^(1/3)

    Parameters:
    -----------
    X, array-like: Input data
    subtract_med, bool: If true, the median of X is substracted before applying the transformation

    Returns:
    --------
    Y, array-like: Transformed data
    """

    if subtract_med:
        Y = X - np.median(X)
    else:
        Y = X

    Y = np.sgn(Y) * np.abs(Y) ** (1. / 3.)


def scale_multimodal(y_transf):
    """Transforms a vector of bi-modal distributed data into a class label and offset.
    """

    means = torch.tensor([y_transf[y_transf < 0.0].mean(), y_transf[y_transf > 0.0].mean()])
    stds = torch.tensor([y_transf[y_transf < 0.0].std(), y_transf[y_transf > 0.0].std()])

    # class_indices = 0: negative, 1: positive
    class_indices = (0.5 * (1. + torch.sign(y_transf))).long()
    delta = (y_transf - means[class_indices]) / stds[class_indices]

    return means, stds, class_indices, delta


class sqrt13_rescaled_subgraph():
    """Defines the sqrt13 transformation and additionally builds subgraphs for
       k-depth neighborhoods."""
    def __init__(self, apar_err_mean: float, apar_err_std: float,
                 dpot_err_mean: float, dpot_err_std: float,
                 apar_res_mean: float, apar_res_std:float,
                 dpot_res_mean:float, dpot_res_std: float,
                 depth: int=2):
        self.apar_err_mean = apar_err_mean
        self.apar_err_std = apar_err_std
        self.dpot_err_mean = dpot_err_mean
        self.dpot_err_std = dpot_err_std
        self.apar_res_mean = apar_res_mean
        self.apar_res_std = apar_res_std
        self.dpot_res_mean = dpot_res_mean
        self.dpot_res_std = dpot_res_std
        self.depth = depth

        print(f"Created scaler: apar(err): {self.apar_err_mean:4.2e} +- {self.apar_err_std:4.2e}")
        print(f"                apar(res): {self.apar_res_mean:4.2e} +- {self.apar_res_std:4.2e}")
        print(f"                dpot(err): {self.dpot_err_mean:4.2e} +- {self.dpot_err_std:4.2e}")
        print(f"                dpot(res): {self.dpot_res_mean:4.2e} +- {self.dpot_res_std:4.2e}")
        print(f"Building subgraphs to depth {self.depth}")


    def __call__(self, data):
        """Applies the pow-1/3 transformation to the last two columns of the x-data
        (apar_res and dpot_res) together with the y-data. Data is subsequently normalized to
        their respective mean (not the residuals) and std.

        Additionally, this builds the k-level neighborhood subgraphs.

        """
        newdata = Data(x=data.x.clone(),
                       edge_attr=data.edge_attr,
                       edge_index=data.edge_index,
                       weight=data.weight,
                       root_vtx=data.root_vtx)

        newdata.x[:, 1:] = torch.sign(newdata.x[:, 1:]) * torch.abs(newdata.x[:, 1:]).pow(1./3.)
        newdata.x[:, -2] = newdata.x[:, -2] / self.apar_res_std
        newdata.x[:, -1] = newdata.x[:, -1] / self.dpot_res_std
        # Transform y to bin center index and offset
        # Scale y-data to order unity
        y_transf = data.y[0, :] * torch.tensor([1e16, 1e8])
        y_transf = torch.sign(y_transf) * torch.abs(y_transf).pow(1./3.)

        # Assume that the distribution is symmetric around 0
        bin_centers = torch.tensor([[-1. * self.apar_err_mean, self.apar_err_mean],
                                    [-1. * self.dpot_err_mean, self.dpot_err_mean]])
        # Class indices are 1 for positive, 0 for negative
        class_indices = (0.5 * (1. + torch.sign(y_transf))).long()
        # Calculate distance to bin center
        delta_y = torch.tensor([(y_transf[0] - bin_centers[0, class_indices[0]]) / self.apar_err_std,
                                (y_transf[1] - bin_centers[1, class_indices[1]]) / self.dpot_err_std])
        newdata.y = torch.unsqueeze(torch.stack([class_indices.double(), delta_y]), 0)

        if self.depth:
            for depth in range(1, self.depth + 1):
                ei_sub, wt_sub = build_subgraph(newdata.edge_index, data.weight, depth=depth)

                setattr(newdata, f"edge_index_{depth:1d}", ei_sub)
                setattr(newdata, f"weight_{depth:1d}", wt_sub)

        return newdata


# End of file transformations.py
