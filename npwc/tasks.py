import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .functions import *
from .modules import *
from .utils import *

def _build_dataset_XOR():
    """
    Builds a dataset and an objective function for XOR problem.
    :return: A tuple (dataset, objective_function).
    """
    def objective_function(model, batch_size, sequence_length):
        """
        Computes the objective function for XOR problem.
        :param model: A model.
        :param batch_size: The batch size.
        :param sequence_length: The sequence length.
        :return: A scalar.
        """
        x = torch.randint(0, 2, (batch_size, sequence_length, 2))
        y = torch.zeros(batch_size, sequence_length, 1)
        for i in range(batch_size):
            for j in range(sequence_length):
                y[i, j, 0] = x[i, j, 0] ^ x[i, j, 1]
        y_hat = model(x)
        return torch.mean(torch.abs(y - y_hat))
    return (None, objective_function)

def _build_dataset_ANDOR():
    """
    Builds a dataset and an objective function for AND/OR problem.
    :return: A tuple (dataset, objective_function).
    """
    def objective_function(model, batch_size, sequence_length):
        """
        Computes the objective function for AND/OR problem.
        :param model: A model.
        :param batch_size: The batch size.
        :param sequence_length: The sequence length.
        :return: A scalar.
        """
        x = torch.randint(0, 2, (batch_size, sequence_length, 2))
        y = torch.zeros(batch_size, sequence_length, 1)
        for i in range(batch_size):
            for j in range(sequence_length):
                y[i, j, 0] = x[i, j, 0] and x[i, j, 1] or x[i, j, 0] or x[i, j, 1]
        y_hat = model(x)
        return torch.mean(torch.abs(y - y_hat))
    return (None, objective_function)

def _build_dataset_SINE_REGRESSION():
    """
    Builds a dataset and an objective function for sine regression problem.
    :return: A tuple (dataset, objective_function).
    """
    def objective_function(model, batch_size, sequence_length):
        """
        Computes the objective function for sine regression problem.
        :param model: A model.
        :param batch_size: The batch size.
        :param sequence_length: The sequence length.
        :return: A scalar.
        """
        x = torch.rand(batch_size, sequence_length, 1)
        y = torch.sin(x)
        y_hat = model(x)
        return torch.mean(torch.abs(y - y_hat))
    return (None, objective_function)

def _build_dataset_COPY():
    """
    Builds a dataset and an objective function for copying task.
    :return: A tuple (dataset, objective_function).
    """
    def objective_function(model, batch_size, sequence_length):
        """
        Computes the objective function for copying task.
        :param model: A model.
        :param batch_size: The batch size.
        :param sequence_length: The sequence length.
        :return: A scalar.
        """
        x = torch.randint(0, 2, (batch_size, sequence_length, 1))
        y = torch.zeros(batch_size, sequence_length, 1)
        for i in range(batch_size):
            for j in range(sequence_length):
                y[i, j, 0] = x[i, j, 0]
        y_hat = model(x)
        return torch.mean(torch.abs(y - y_hat))
    return (None, objective_function)

def _build_dataset_LINKED_LIST():
    """
    Builds a dataset and an objective function for linked list task.
    :return: A tuple (dataset, objective_function).
    """
    def objective_function(model, batch_size, sequence_length):
        """
        Computes the objective function for linked list task.
        :param model: A model.
        :param batch_size: The batch size.
        :param sequence_length: The sequence length.
        :return: A scalar.
        """
        x = torch.randint(0, 2, (batch_size, sequence_length, 1))
        y = torch.zeros(batch_size, sequence_length, 1)
        for i in range(batch_size):
            for j in range(sequence_length):
                if j == 0:
                    y[i, j, 0] = x[i, j, 0]
                else:
                    y[i, j, 0] = x[i, j, 0] and y[i, j - 1, 0]
        y_hat = model(x)
        return torch.mean(torch.abs(y - y_hat))
    return (None, objective_function)

def _build_dataset_RECURSION():
    """
    Builds a dataset and an objective function for recursion task.
    :return: A tuple (dataset, objective_function).
    """
    def objective_function(model, batch_size, sequence_length):
        """
        Computes the objective function for recursion task.
        :param model: A model.
        :param batch_size: The batch size.
        :param sequence_length: The sequence length.
        :return: A scalar.
        """
        x = torch.randint(0, 2, (batch_size, sequence_length, 1))
        y = torch.zeros(batch_size, sequence_length, 1)
        for i in range(batch_size):
            for j in range(sequence_length):
                if j == 0:
                    y[i, j, 0] = x[i, j, 0]
                else:
                    y[i, j, 0] = x[i, j, 0] and y[i, j - 1, 0]
        y_hat = model(x)
        return torch.mean(torch.abs(y - y_hat))
    return (None, objective_function)

def _build_dataset_MERGE_SORT():
    """
    Builds a dataset and an objective function for merge sort task.
    :return: A tuple (dataset, objective_function).
    """
    def objective_function(model, batch_size, sequence_length):
        """
        Computes the objective function for merge sort task.
        :param model: A model.
        :param batch_size: The batch size.
        :param sequence_length: The sequence length.
        :return: A scalar.
        """
        x = torch.randint(0, 2, (batch_size, sequence_length, 1))
        y = torch.zeros(batch_size, sequence_length, 1)
        for i in range(batch_size):
            for j in range(sequence_length):
                if j == 0:
                    y[i, j, 0] = x[i, j, 0]
                else:
                    y[i, j, 0] = x[i, j, 0] and y[i, j - 1, 0]
        y_hat = model(x)
        return torch.mean(torch.abs(y - y_hat))
    return (None, objective_function)
