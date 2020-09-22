import random
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def edge_perms(l, window_past, window_future):
    """
    Method to construct the edges considering the past and future window.
    用于构造边的函数
    """
    all_perms = set()
    array = np.arange(l)
    for j in range(l):
        perms = set()

        if window_past == -1 and window_future == -1:
            eff_array = array
        elif window_past == -1:
            eff_array = array[:min(l, j + window_future + 1)]
        elif window_future == -1:
            eff_array = array[max(0, j - window_past):]
        else:
            eff_array = array[max(0, j - window_past):min(l, j + window_future + 1)]

        for item in eff_array:
            perms.add((j, item))
        all_perms = all_perms.union(perms)
    return list(all_perms)


def hgt_batch_graphify(features, qmask, lengths, window_past, window_future, no_cuda):
    """
    Construct batch graphs for HGT model, mainly adding node types.
    """
    edge_index, edge_type, node_type, node_features = [], [], [], []
    batch_size = features.size(1)
    length_sum = 0

    edge_index_lengths = []

    # TODO: No need to add edge weights?
    # edge_ind = []
    # for j in range(batch_size):  # Add edges -- mainly for calculating edge weights
    #     edge_ind.append(edge_perms(lengths[j], window_past, window_future)
    # scores are the edge weights!!!
    # scores = att_model(features, lengths, edge_ind)

    for j in range(batch_size):
        node_features.append(features[:lengths[j], j, :])          # Add node features to graph.
        node_type.extend((qmask[:lengths[j], j, :] == 1).nonzero()[:, 1].tolist())
        perms1 = edge_perms(lengths[j], window_past, window_future)
        perms2 = [(item[0] + length_sum, item[1] + length_sum) for item in perms1]
        length_sum += lengths[j]

        edge_index_lengths.append(len(perms1))

        for item1, item2 in zip(perms1, perms2):
            edge_index.append(torch.tensor([item2[0], item2[1]]))  # Add edges to graph.
            if item1[0] < item1[1]:                                # Add edge types to graph.
                edge_type.append(0)  # future
            else:
                edge_type.append(1)  # past

    node_features = torch.cat(node_features, dim=0)
    edge_index = torch.stack(edge_index).transpose(0, 1)
    edge_type = torch.tensor(edge_type)
    node_type = torch.tensor(node_type)
    
    if not no_cuda:
        node_features = node_features.cuda()
        node_type = node_type.cuda()
        edge_index = edge_index.cuda()
        edge_type = edge_type.cuda()

    return node_features, node_type, edge_index, edge_type, edge_index_lengths
