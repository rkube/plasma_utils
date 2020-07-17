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

def zero_filter(data, eps=1e-22):
    """Discard the data object if the target values are below a threshold."""
    return ~torch.any(torch.abs(data.y) < eps)


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


class subgraphs():
    """Adds k-depth neighborhoods to a data object."""
    def __init__(self, depth):
        self.depth = depth

    def __call__(self, data):
        """Builds k-depth subgraphs appends them to the data structure."""

        for depth in range(1, self.depth + 1):
            ei_sub, wt_sub = build_subgraph(data.edge_index, data.weight, depth=depth)

            setattr(data, f"edge_index_{depth:1d}", ei_sub)
            setattr(data, f"weight_{depth:1d}", wt_sub)

        return data


class apply_transform_x():
    """Applies a transformation to the data."""

    def __init__(self, trf, idx):
        self.trf = trf
        self.idx = idx
        self.factor = None
        if self.idx == 7:
            self.factor = 1e13
        elif self.idx == 8:
            self.factor = 1e7
        else:
            self.factor = 1.0

    def __call__(self, data_orig):

        new_data = Data(x=data_orig.x.clone().detach(),
                        y=data_orig.y.clone().detach(),
                        edge_index=data_orig.edge_index.clone().detach(),
                        edge_attr=data_orig.edge_attr.clone().detach(),
                        root_vtx=data_orig.root_vtx)
        values = np.expand_dims(data_orig.x[:, self.idx].numpy() * self.factor, 1)
        new_data.x[:, self.idx] = torch.as_tensor(self.trf.transform(values).squeeze())
        return new_data


class apply_transform_y():
    """Applies transformation for regression target.
    Transform both regression targets simultaneously so that we can clip the target
    vector size to only include the root-vertex target."""

    def __init__(self, trf_y0, trf_y1):
        self.trf_y0 = trf_y0
        self.trf_y1 = trf_y1

    def __call__(self, data_orig):
        # Get regression targets for the root vertex, multiply by 1e13/1e7 and
        # apply quantile transformation
        new_data = Data(x=data_orig.x.clone().detach(),
                        edge_index=data_orig.edge_index.clone().detach(),
                        edge_attr=data_orig.edge_attr.clone().detach(),
                        root_vtx=data_orig.root_vtx)
        values = data_orig.y[0, :].numpy()
        y0 = self.trf_y0.transform((values[0] * 1e16).reshape(1, -1)).item()
        y1 = self.trf_y1.transform((values[1] * 1e8).reshape(1, -1)).item()
        new_data.y = torch.tensor([[y0, y1]])

        return new_data


class apply_transform():
    """Applies a transformation to the data."""

    def __init__(self, trf_x_list, idx_x_list, trf_y_list):
        self.trf_x_list = trf_x_list
        self.idx_x_list = idx_x_list
        self.trf_y_list = trf_y_list


    def __call__(self, data_orig):

        new_data_x = torch.zeros_like(data_orig.x)
        new_data_y = torch.zeros_like(data_orig.y)

        for trf, idx in zip(self.trf_x_list, self.idx_x_list):
            values = np.expand_dims(data_orig.x[:, idx].numpy(), 1)
            new_data_x[:, idx] = torch.as_tensor(trf.transform(values).squeeze())

        for trf, idx in zip(self.trf_y_list, [0, 1]):
            values = np.expand_dims(data_orig.y[:, idx].numpy(), 1)
            new_data_y[:, idx] = torch.as_tensor(trf.transform(values).squeeze())

        new_data = Data(x=new_data_x,
                        y=new_data_y,
                        edge_index=data_orig.edge_index.clone().detach(),
                        root_vtx=data_orig.root_vtx)

        return new_data


# End of file transformations.py
