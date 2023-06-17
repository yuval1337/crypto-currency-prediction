import torch
import torch.nn as nn
from .typing import *


class CryptoPredictorModel(Module):
  '''Implements a custom hybrid CRNN deep-learning model.

  The model will be used to predict a cryptocurrency's value over time.
  '''

  def __init__(self):
    '''C'tor.

    A definition of this model's architecture will be described below.
    '''
    ...

  def forward(self, x: Tensor) -> Tensor:
    '''Defines forward pass implementation in this model's neural network.

    Args:
      x (Tensor): Some Tensor to pass through the network.

    Returns:
      Tensor: This model's prediction for the input 'x'.
    '''
    ...
