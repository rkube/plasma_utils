# -*- Encoding: UTF-8 -*-

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU
from torch_geometric.nn import  CGConv

class CGNet_flatten_GRU(torch.nn.Module):
    """Flattened CGNet that feeds data from all depth neighborhoods through the
       same convolutional layer."""
    def __init__(self, n_features: int, n_edge:int, conv_dim:int=64, depth:int=2):
        super(CGNet_flatten_GRU, self).__init__()

        self.depth = depth

        self.fc1 = torch.nn.Linear(9, conv_dim)

        # OBS: root_weight needs to be false. We already account for by including a self-loop
        # to the root vertex in edge_index
        self.conv_l1 = CGConv(conv_dim, 3, aggr="mean", flow="target_to_source")
        self.gru_l1 = GRU(conv_dim, conv_dim)

        self.fc_shared = torch.nn.Linear(conv_dim, conv_dim)

        self.fc2_class = torch.nn.Linear(conv_dim, 64)
        self.fc3_class = torch.nn.Linear(64, 2)

        self.fc2_delta = torch.nn.Linear(conv_dim, 64)
        self.fc3_delta = torch.nn.Linear(64, 2)

    def forward(self, data):
        #batch_g = data.batch.to(device="cuda")

        x = F.relu(self.fc1(data.x))
        x = F.dropout(x, p=0.15, training=self.training)
        h = x.unsqueeze(0)

        for i in range(3):
            if self.depth == 5:
                m = F.relu(self.conv_l1(x, data.edge_index_5, data.weight_5))
                m = F.relu(self.conv_l1(m, data.edge_index_4, data.weight_4))
                m = F.relu(self.conv_l1(m, data.edge_index_3, data.weight_3))
                m = F.relu(self.conv_l1(m, data.edge_index_2, data.weight_2))
                m = F.relu(self.conv_l1(m, data.edge_index_1, data.weight_1))
            elif self.depth == 4:
                m = F.relu(self.conv_l1(x, data.edge_index_4, data.weight_4))
                m = F.relu(self.conv_l1(m, data.edge_index_3, data.weight_3))
                m = F.relu(self.conv_l1(m, data.edge_index_2, data.weight_2))
                m = F.relu(self.conv_l1(m, data.edge_index_1, data.weight_1))
            elif self.depth == 3:
                m = F.relu(self.conv_l1(x, data.edge_index_3, data.weight_3))
                m = F.relu(self.conv_l1(m, data.edge_index_2, data.weight_2))
                m = F.relu(self.conv_l1(m, data.edge_index_1, data.weight_1))
            elif self.depth == 2:
                m = F.relu(self.conv_l1(x, data.edge_index_2, data.weight_2))
                m = F.relu(self.conv_l1(m, data.edge_index_1, data.weight_1))
            elif self.depth == 1:
                m = F.relu(self.conv_l1(x, data.edge_index_1, data.weight_1))
            else:
                raise ValueError("We need to hvae 1 <= depth <= 5")


            x, h = self.gru_l1(m.unsqueeze(0), h)
            x = x.squeeze(0)

        # For the dataset at hand, the index of the center node is always at the beginning
        # of the batch. Find the index where each unique element in the batch occurs first
        # and use this as an index array to get the new features at the center node
        idx_ct = torch.cat([(data.batch == u).nonzero()[0] for u in torch.unique(data.batch)])
        x = x[idx_ct, :]

        # x is the convolution over the entire graphlet.
        # We only need to forward the x-values at the center nodes.
        x = F.relu(self.fc_shared(x))

        x_class = F.relu(self.fc2_class(x))
        x_class = F.dropout(x_class, p=0.15, training=self.training)
        x_class = torch.sigmoid(self.fc3_class(x_class))

        x_delta = F.relu(self.fc2_delta(x))
        x_delta = F.dropout(x_delta, p=0.15, training=self.training)
        x_delta = self.fc3_delta(x_delta)

        return(x_class, x_delta)



class CGNet_flatten(torch.nn.Module):
    """Flattened CGNet that feeds data from all depth neighborhoods through the
       same convolutional layer."""
    def __init__(self, n_features: int, n_edge:int, conv_dim:int=64, depth:int=2):
        super(CGNet_flatten, self).__init__()

        self.depth = depth

        self.fc1 = torch.nn.Linear(9, conv_dim)

        # OBS: root_weight needs to be false. We already account for by including a self-loop
        # to the root vertex in edge_index
        self.conv_l1 = CGConv(conv_dim, 3, aggr="mean", flow="target_to_source")

        self.fc_shared = torch.nn.Linear(conv_dim, conv_dim)

        self.fc2_class = torch.nn.Linear(conv_dim, 64)
        self.fc3_class = torch.nn.Linear(64, 2)

        self.fc2_delta = torch.nn.Linear(conv_dim, 64)
        self.fc3_delta = torch.nn.Linear(64, 2)

    def forward(self, data):
        #batch_g = data.batch.to(device="cuda")

        x = F.relu(self.fc1(data.x))
        x = F.dropout(x, p=0.15, training=self.training)

        if self.depth == 5:
            x = F.relu(self.conv_l1(x, data.edge_index_5, data.weight_5))
            x = F.relu(self.conv_l1(x, data.edge_index_4, data.weight_4))
            x = F.relu(self.conv_l1(x, data.edge_index_3, data.weight_3))
            x = F.relu(self.conv_l1(x, data.edge_index_2, data.weight_2))
            x = F.relu(self.conv_l1(x, data.edge_index_1, data.weight_1))
        elif self.depth == 4:
            x = F.relu(self.conv_l1(x, data.edge_index_4, data.weight_4))
            x = F.relu(self.conv_l1(x, data.edge_index_3, data.weight_3))
            x = F.relu(self.conv_l1(x, data.edge_index_2, data.weight_2))
            x = F.relu(self.conv_l1(x, data.edge_index_1, data.weight_1))
        elif self.depth == 3:
            x = F.relu(self.conv_l1(x, data.edge_index_3, data.weight_3))
            x = F.relu(self.conv_l1(x, data.edge_index_2, data.weight_2))
            x = F.relu(self.conv_l1(x, data.edge_index_1, data.weight_1))
        elif self.depth == 2:
            x = F.relu(self.conv_l1(x, data.edge_index_2, data.weight_2))
            x = F.relu(self.conv_l1(x, data.edge_index_1, data.weight_1))
        elif self.depth == 1:
            x = F.relu(self.conv_l1(x, data.edge_index_1, data.weight_1))
        else:
                raise ValueError("We need to have 1 <= depth <= 5")

        # For the dataset at hand, the index of the center node is always at the beginning
        # of the batch. Find the index where each unique element in the batch occurs first
        # and use this as an index array to get the new features at the center node
        idx_ct = torch.cat([(data.batch == u).nonzero()[0] for u in torch.unique(data.batch)])
        x = x[idx_ct, :]

        # x is the convolution over the entire graphlet.
        # We only need to forward the x-values at the center nodes.
        x = F.relu(self.fc_shared(x))

        x_class = F.relu(self.fc2_class(x))
        x_class = F.dropout(x_class, p=0.15, training=self.training)
        x_class = torch.sigmoid(self.fc3_class(x_class))

        x_delta = F.relu(self.fc2_delta(x))
        x_delta = F.dropout(x_delta, p=0.15, training=self.training)
        x_delta = self.fc3_delta(x_delta)

        return(x_class, x_delta)


class CGNet_GRU_plain_01(torch.nn.Module):
    """Flattened CGNet that feeds data from all depth neighborhoods through the
       same convolutional layer."""
    def __init__(self, n_features: int, n_edge:int, conv_dim:int=64, depth:int=2):
        super(CGNet_GRU_plain_01, self).__init__()

        self.depth = depth

        self.fc1 = torch.nn.Linear(9, conv_dim)

        # OBS: root_weight needs to be false. We already account for by including a self-loop
        # to the root vertex in edge_index
        self.conv_l1 = CGConv(conv_dim, 3, aggr="mean", flow="target_to_source")
        self.gru_l1 = GRU(conv_dim, conv_dim)

        self.fc2 = torch.nn.Linear(conv_dim, conv_dim)

        self.fc3 = torch.nn.Linear(conv_dim, 64)
        self.fc4 = torch.nn.Linear(64, 2)

    def forward(self, data):
        #batch_g = data.batch.to(device="cuda")

        x = F.relu(self.fc1(data.x))
        x = F.dropout(x, p=0.15, training=self.training)
        h = x.unsqueeze(0)

        for i in range(1):
            if self.depth == 5:
                m = F.relu(self.conv_l1(x, data.edge_index_5, data.weight_5))
                m = F.relu(self.conv_l1(m, data.edge_index_4, data.weight_4))
                m = F.relu(self.conv_l1(m, data.edge_index_3, data.weight_3))
                m = F.relu(self.conv_l1(m, data.edge_index_2, data.weight_2))
                m = F.relu(self.conv_l1(m, data.edge_index_1, data.weight_1))
            elif self.depth == 4:
                m = F.relu(self.conv_l1(x, data.edge_index_4, data.weight_4))
                m = F.relu(self.conv_l1(m, data.edge_index_3, data.weight_3))
                m = F.relu(self.conv_l1(m, data.edge_index_2, data.weight_2))
                m = F.relu(self.conv_l1(m, data.edge_index_1, data.weight_1))
            elif self.depth == 3:
                m = F.relu(self.conv_l1(x, data.edge_index_3, data.weight_3))
                m = F.relu(self.conv_l1(m, data.edge_index_2, data.weight_2))
                m = F.relu(self.conv_l1(m, data.edge_index_1, data.weight_1))
            elif self.depth == 2:
                m = F.relu(self.conv_l1(x, data.edge_index_2, data.weight_2))
                m = F.relu(self.conv_l1(m, data.edge_index_1, data.weight_1))
            elif self.depth == 1:
                m = F.relu(self.conv_l1(x, data.edge_index_1, data.weight_1))
            else:
                raise ValueError("We need to hvae 1 <= depth <= 5")


            x, h = self.gru_l1(m.unsqueeze(0), h)
            x = x.squeeze(0)

        # For the dataset at hand, the index of the center node is always at the beginning
        # of the batch. Find the index where each unique element in the batch occurs first
        # and use this as an index array to get the new features at the center node
        idx_ct = torch.cat([(data.batch == u).nonzero()[0] for u in torch.unique(data.batch)])
        x = x[idx_ct, :]

        # x is the convolution over the entire graphlet.
        # We only need to forward the x-values at the center nodes.
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        x = self.fc4(x)

        return(x)

class CGNet_squashed(torch.nn.Module):
    """Flattened CGNet that feeds data from all depth neighborhoods through the
       same convolutional layer."""
    def __init__(self, n_features: int, n_edge:int, conv_dim:int=64):
        super(CGNet_squashed, self).__init__()

        self.fc1 = torch.nn.Linear(9, conv_dim)

        # OBS: root_weight needs to be false. We already account for by including a self-loop
        # to the root vertex in edge_index
        self.conv_l1 = CGConv(conv_dim, n_edge, aggr="mean", flow="target_to_source")

        self.fc2 = torch.nn.Linear(conv_dim, conv_dim)

        self.fc3 = torch.nn.Linear(conv_dim, 64)
        self.fc4 = torch.nn.Linear(64, 2)

    def forward(self, data):
        #batch_g = data.batch.to(device="cuda")

        x = F.relu(self.fc1(data.x))
        x = F.dropout(x, p=0.15, training=self.training)

        x = self.conv_l1(x, data.edge_index, data.edge_attr)
        x = F.relu(x)

        # For the dataset at hand, the index of the center node is always at the beginning
        # of the batch. Find the index where each unique element in the batch occurs first
        # and use this as an index array to get the new features at the center node
        idx_ct = torch.cat([(data.batch == u).nonzero()[0] for u in torch.unique(data.batch)])
        x = x[idx_ct, :]

        # x is the convolution over the entire graphlet.
        # We only need to forward the x-values at the center nodes.
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))


        x = self.fc4(x)

        return(x)


class CGNet_flatten_GRU_v02(torch.nn.Module):
    """This is the version used with a refactored version of the ML data loading
    routines. That is the ones using routines
    * load_simulation_data
    * StandardScalers
    * zero_filter
    * my_scaler

    """Flattened CGNet that feeds data from all depth neighborhoods through the
       same convolutional layer."""
    def __init__(self, n_features: int, n_edge:int, conv_dim:int=64):
        super(CGNet_flatten_GRU, self).__init__()

        #self.depth = depth

        self.fc1 = torch.nn.Linear(9, conv_dim)

        # OBS: root_weight needs to be false. We already account for by including a self-loop
        # to the root vertex in edge_index
        self.conv_l1 = CGConv(conv_dim, n_edge, aggr="mean", flow="target_to_source")

        self.gru_l1 = GRU(conv_dim, conv_dim)

        self.fc2 = torch.nn.Linear(conv_dim, conv_dim)

        self.fc3 = torch.nn.Linear(conv_dim, 64)
        self.fc4 = torch.nn.Linear(64, 2)

    def forward(self, data, root_offset):
        """
        Parameters:
        -----------
        data........: batch with tensor x: dim0: num samples, dim
        root_offset.: indices where the root nodes are located in the current batch
        """
        x = F.relu(self.fc1(data.x))
        x = F.dropout(x, p=0.15, training=self.training)
        h = x.unsqueeze(0)

        for i in range(3):
            m = self.conv_l1(x, data.edge_index, data.edge_attr)
            x, h = self.gru_l1(m.unsqueeze(0), h)
            x = x.squeeze(0)
        # x is the convolution over the entire graphlet.
        # We only need to forward the x-values at the center nodes.

        x = F.relu(x[root_offset, :])

        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))


        x = self.fc4(x)

        return(x)


# End of file models_cgconv.py
