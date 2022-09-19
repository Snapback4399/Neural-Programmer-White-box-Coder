import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .functions import *
from .modules import *
from .tasks import *

def image_grid(tensors, rows = -1, cols = 5, border = 1):
    """
    Given a list of tensors l, generate a single such that if l has shape (BATCH_SIZE, D, H, W) then the result has shape (cols * H + border, rows * W + border) and the batch is tiled along the first dimension so that the output looks like an image grid. It is assumed that all tensors in l have the same shape.
    :param tensors: A list of tensors.
    :param rows: The number of rows.
    :param cols: The number of columns.
    :param border: The border width.
    :return: A tensor of shape (cols * H + border, rows * W + border).
    """
    if rows == -1:
        rows = len(tensors) // cols
    if len(tensors) % cols != 0:
        rows += 1
    height = tensors[0].shape[2]
    width = tensors[0].shape[3]
    result = torch.zeros(rows * height + border, cols * width + border)
    for i in range(rows):
        for j in range(cols):
            if i * cols + j < len(tensors):
                result[i * height + border:(i + 1) * height + border, j * width + border:(j + 1) * width + border] = tensors[i * cols + j]
    return result

def print_model(label, model_desc):
    """
    Displays any model either as a graph or as a flowchart depending on the settings.
    :param label: The label of the model.
    :param model_desc: The description of the model.
    """
    if 'graph' in model_desc:
        print('{}:'.format(label))
        print(model_desc['graph'])
    elif 'flowchart' in model_desc:
        print('{}:'.format(label))
        plot_layout(model_desc['flowchart']['obs'], model_desc['flowchart']['layout'])
