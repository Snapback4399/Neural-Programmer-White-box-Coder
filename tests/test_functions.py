import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from npwc.functions import *

class TestFunctions(unittest.TestCase):
    def test_one_hot(self):
        indices = torch.tensor([[0, 1, 2], [3, 4, 5]])
        depth = 6
        one_hot_matrix = one_hot(indices, depth)
        self.assertEqual(one_hot_matrix.shape, (2, 6, 3))
        self.assertTrue(torch.all(one_hot_matrix[0, 0, :] == torch.tensor([1., 0., 0.])))
        self.assertTrue(torch.all(one_hot_matrix[0, 1, :] == torch.tensor([0., 1., 0.])))
        self.assertTrue(torch.all(one_hot_matrix[0, 2, :] == torch.tensor([0., 0., 1.])))
        self.assertTrue(torch.all(one_hot_matrix[0, 3, :] == torch.tensor([0., 0., 0.])))
        self.assertTrue(torch.all(one_hot_matrix[0, 4, :] == torch.tensor([0., 0., 0.])))
        self.assertTrue(torch.all(one_hot_matrix[0, 5, :] == torch.tensor([0., 0., 0.])))
        self.assertTrue(torch.all(one_hot_matrix[1, 0, :] == torch.tensor([0., 0., 0.])))
        self.assertTrue(torch.all(one_hot_matrix[1, 1, :] == torch.tensor([0., 0., 0.])))
        self.assertTrue(torch.all(one_hot_matrix[1, 2, :] == torch.tensor([0., 0., 0.])))
        self.assertTrue(torch.all(one_hot_matrix[1, 3, :] == torch.tensor([1., 0., 0.])))
        self.assertTrue(torch.all(one_hot_matrix[1, 4, :] == torch.tensor([0., 1., 0.])))
        self.assertTrue(torch.all(one_hot_matrix[1, 5, :] == torch.tensor([0., 0., 1.])))

    def test_locally_connected_matmul(self):
        a = torch.tensor([[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]])
        b = torch.tensor([[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]])
        c = _locally_connected_matmul(a, b)
        self.assertEqual(c.shape, (1, 1, 2, 2))
        self.assertTrue(torch.all(c[0, 0, 0, 0] == torch.tensor([7., 10.])))
        self.assertTrue(torch.all(c[0, 0, 0, 1] == torch.tensor([15., 22.])))
        self.assertTrue(torch.all(c[0, 0, 1, 0] == torch.tensor([23., 34.])))
        self.assertTrue(torch.all(c[0, 0, 1, 1] == torch.tensor([31., 46.])))
