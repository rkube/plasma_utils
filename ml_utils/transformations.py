# -*- Encoding: UTF-8 -*-

import numpy as np

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


class sqrt13_rescaled():
    def __init__(self, apar_err_mean, apar_err_std, dpot_err_mean, dpot_err_std,
                 apar_res_mean, apar_res_std, dpot_res_mean, dpot_res_std):
        self.apar_err_mean = apar_err_mean
        self.apar_err_std = apar_err_std
        self.dpot_err_mean = dpot_err_mean
        self.dpot_err_std = dpot_err_std
        self.apar_res_mean = apar_res_mean
        self.apar_res_std = apar_res_std
        self.dpot_res_mean = dpot_res_mean
        self.dpot_res_std = dpot_res_std
        
        print(f"Created scaler. apar: {self.apar_mean:4.2e}+-{self.apar_std:4.2e}, dpot: {self.dpot_mean:4.2e}+-{self.dpot_std:4.2e}")
        
        
    def __call__(self, data):
       
        newdata = Data(x=data.x.clone(), 
                       edge_attr=data.edge_attr, 
                       edge_index=data.edge_index)
        
        newdata.x[:, 1:] = torch.sign(newdata.x[:, 1:]) * torch.abs(newdata.x[:, 1:]).pow(1./3.) 
        # Transform y to bin center index and offset
        # Scale y-data to order unity   
        y_transf = data.y * torch.tensor([1e16, 1e8])
        y_transf = torch.sign(y_transf) * torch.abs(y_transf).pow(1./3.)

        # Assume that the distribution is symmetric around 0
        bin_centers = torch.tensor([[-1. * self.apar_mean, self.apar_mean], 
                                    [-1. * self.dpot_mean, self.dpot_mean]])
        # Class indices are 1 for positive, 0 for negative
        class_indices = (0.5 * (1. + torch.sign(y_transf))).long()
        # Calculate distance to bin center
        delta_y = torch.tensor([(y_transf[0] - bin_centers[0, class_indices[0]]) / self.apar_std,
                                (y_transf[1] - bin_centers[1, class_indices[1]]) / self.dpot_std])
        newdata.y = torch.unsqueeze(torch.stack([class_indices.double(), delta_y]), 0)

        return newdata   


# End of file transformations.py