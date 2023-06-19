import torch
import torch.nn as nn
from .typing import *


class CryptoPredictorModel(Module):
  '''Implements a custom hybrid CRNN deep-learning model.

  The model will be used to predict a cryptocurrency's value over time.
  '''

  def __init__(self, input_size: int, hidden_size: int, output_size: int):
    '''C'tor.

    A definition of this model's architecture will be described below.
    '''
    super(CryptoPredictorModel, self).__init__()
    self.hidden_size = hidden_size
    self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x: Tensor) -> Tensor:
    '''Defines forward pass implementation in this model's neural network.

    Args:
      x (Tensor): Some Tensor to pass through the network.

    Returns:
      Tensor: This model's prediction for the input 'x'.
    '''
    out, _ = self.lstm(x)
    out = self.fc(out[:, -1, :])  # Get the last output from LSTM sequence
    return out
