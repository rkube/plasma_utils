#-*- Encoding: UTF-8 -*-

import torch
import torch.nn.functional as F


def train_reg(loader, optimizer, model):
    """Training for the regression-only loss.

    Parameters:
    -----------
    loader: torch_geometric.data.DataLoader
    optimizer: torch.optim
    model: torch.nn

    Returns:
    --------
    float, loss for current batch
    """
    model.train()
    loss_all = 0.0

    for data in loader:
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, data.y, reduction="mean")
        loss.backward()
        optimizer.step()

        loss_all += loss.item()

    return(loss_all / len(loader))


def validate_reg(loader, model):
    """Calculates avg. cross-entropy loss for classification and L2 loss for regression.

    Parameters:
    -----------
    loader: torch_geometric.data.DataLoader
    model: torch.nn

    Returns:
    --------
    float, loss of current batch
    """

    model.eval()
    loss_all = 0.0
    for data in loader:
        with torch.no_grad():
            out = model(data)
            loss = F.mse_loss(out, data.y, reduction="mean")
            loss_all += loss.item()

    return(loss_all / len(loader))


def train_class_reg(loader, optimizer, model, device, alpha:float=1.0):
    """Training for the split classification/regression model.

    Parameters:
    -----------
    loader: torch_geometric.data.DataLoader
    optimizer: torch.optim
    model: torch.nn
    alpha: multiplier for classification loss

    Returns:
    --------
    Tuple, (Avg. cross-entropy loss for classification per graph, Avg. L2 loss per graph)
    """
    model.train()
    loss_class_all = 0.0
    loss_delta_all = 0.0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out_class, out_delta = model(data)

        loss_class = F.binary_cross_entropy(out_class, data.y[:,0,:], reduction="mean")
        loss_delta = F.mse_loss(out_delta, data.y[:,1,:], reduction="mean")

        total_loss = alpha * loss_class + loss_delta
        total_loss.backward()
        optimizer.step()

        loss_class_all += loss_class.item()
        loss_delta_all += loss_delta.item()

    return(alpha * loss_class_all / len(loader), loss_delta_all / len(loader))

def validate_class_reg(loader, model, device, alpha: float=1.0):
    """Calculates avg. cross-entropy loss for classification and L2 loss for regression.

    Parameters:
    -----------
    loader: torch_geometric.data.DataLoader
    model: torch.nn
    alpha: multiplier for classification loss

    Returns:
    --------
    Tuple, (Avg. cross-entropy loss for classification per graph, Avg. L2 loss per graph)
    """
    model.eval()
    loss_class_all = 0.0
    loss_delta_all = 0.0
    for data in loader:
        data = data.to(device)

        with torch.no_grad():
            out_class, out_delta = model(data.to(device))

            loss_class = F.binary_cross_entropy(out_class, data.y[:,0,:], reduction="mean")
            loss_delta = F.mse_loss(out_delta, data.y[:,1,:], reduction="mean")

            loss_class_all += loss_class.item()
            loss_delta_all += loss_delta.item()

    return(alpha * loss_class_all / len(loader), loss_delta_all / len(loader))

# End of file misc.py
