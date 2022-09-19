import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def _validate_parameter(function, parameter_name, parameter_value):
    if not isinstance(parameter_value, int):
        raise ValueError("{}() argument '{}' must be an integer".format(function, parameter_name))
    if parameter_value < 0:
        raise ValueError("{}() argument '{}' must be non-negative".format(function, parameter_name))

def one_hot(indices, depth):
    """
    One-hot encode an integer tensor in the way that converts it to a matrix of binary values.
    :param indices: A tensor of indices.
    :param depth: The number of classes.
    :return: A tensor of shape (BATCH_SIZE, depth, max_index + 1) where max_index is the maximum value in indices.
    """
    _validate_parameter('one_hot', 'depth', depth)
    max_index = torch.max(indices)
    one_hot_matrix = torch.zeros(indices.shape[0], depth, max_index + 1)
    for i in range(indices.shape[0]):
        for j in range(indices.shape[1]):
            one_hot_matrix[i, indices[i, j], j] = 1.
    return one_hot_matrix

def _locally_connected_matmul(a, b):
    """
    Multiplies two arrays using a locally connected approach (a variant of convolution).
    :param a: A tensor of shape (BATCH_SIZE, D, H, W).
    :param b: A tensor of shape (D, H, W, K).
    :return: A tensor of shape (BATCH_SIZE, K, H, W).
    """
    assert a.shape[1] == b.shape[0]
    assert a.shape[2] == b.shape[1]
    assert a.shape[3] == b.shape[2]
    result = torch.zeros(a.shape[0], b.shape[3], a.shape[2], a.shape[3])
    for i in range(a.shape[0]):
        for j in range(b.shape[3]):
            for k in range(a.shape[1]):
                result[i, j] += torch.mul(a[i, k], b[k, :, :, j])
    return result
