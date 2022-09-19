import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .functions import *

class LookupTableModule(nn.Module):
    """
    A lookup table module.
    """
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout):
        """
        Initializes a lookup table module.
        :param input_size: The size of the input.
        :param output_size: The size of the output.
        :param hidden_size: The size of the hidden state.
        :param num_layers: The number of layers.
        :param dropout: The dropout probability.
        """
        super(LookupTableModule, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Runs the module.
        :param x: A tensor of shape (BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE).
        :return: A tensor of shape (BATCH_SIZE, SEQUENCE_LENGTH, OUTPUT_SIZE).
        """
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out)
        return out

class CompositionModule(nn.Module):
    """
    A module for function composition.
    """
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout):
        """
        Initializes a composition module.
        :param input_size: The size of the input.
        :param output_size: The size of the output.
        :param hidden_size: The size of the hidden state.
        :param num_layers: The number of layers.
        :param dropout: The dropout probability.
        """
        super(CompositionModule, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Runs the module.
        :param x: A tensor of shape (BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE).
        :return: A tensor of shape (BATCH_SIZE, SEQUENCE_LENGTH, OUTPUT_SIZE).
        """
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out)
        return out

class ConcatModule(nn.Module):
    """
    A module that concatenates output on all the input sequences of a module into a single vector.
    """
    def __init__(self, module):
        """
        Initializes a concat module.
        :param module: The module to be wrapped.
        """
        super(ConcatModule, self).__init__()
        self.module = module

    def forward(self, x):
        """
        Runs the module.
        :param x: A tensor of shape (BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE).
        :return: A tensor of shape (BATCH_SIZE, OUTPUT_SIZE).
        """
        out = self.module(x)
        out = torch.reshape(out, (out.shape[0], out.shape[1] * out.shape[2]))
        return out

class SumModule(nn.Module):
    """
    A module for summation of vectors.
    """
    def __init__(self, module):
        """
        Initializes a sum module.
        :param module: The module to be wrapped.
        """
        super(SumModule, self).__init__()
        self.module = module

    def forward(self, x):
        """
        Runs the module.
        :param x: A tensor of shape (BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE).
        :return: A tensor of shape (BATCH_SIZE, OUTPUT_SIZE).
        """
        out = self.module(x)
        out = torch.sum(out, dim=1)
        return out

class RNNModule(nn.Module):
    """
    A recurrent neural network module.
    """
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout):
        """
        Initializes an RNN module.
        :param input_size: The size of the input.
        :param output_size: The size of the output.
        :param hidden_size: The size of the hidden state.
        :param num_layers: The number of layers.
        :param dropout: The dropout probability.
        """
        super(RNNModule, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Runs the module.
        :param x: A tensor of shape (BATCH_SIZE, SEQUENCE_LENGTH, INPUT_SIZE).
        :return: A tensor of shape (BATCH_SIZE, SEQUENCE_LENGTH, OUTPUT_SIZE).
        """
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out)
        return out

def empty_module():
    """
    Returns a module which gives a constant output of 1.
    :return: A module.
    """
    class EmptyModule(nn.Module):
        def forward(self, x):
            return torch.ones(x.shape[0], x.shape[1], 1)
    return EmptyModule()
