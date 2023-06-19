import torch
import torch.nn as nn
from .typing import *


# class CryptoPredictorModel(Module):
#   '''Implements a custom hybrid CRNN deep-learning model.

#   The model will be used to predict a cryptocurrency's value over time.
#   '''

#   def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 1):
#     '''C'tor.

#     A definition of this model's architecture will be described below.
#     '''
#     super(CryptoPredictorModel, self).__init__()
#     self.hidden_size = hidden_size
#     self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
#     self.fc = nn.Linear(hidden_size, output_size)

#   def forward(self, x: Tensor) -> Tensor:
#     '''Defines forward pass implementation in this model's neural network.

#     Args:
#       x (Tensor): Some Tensor to pass through the network.

#     Returns:
#       Tensor: This model's prediction for the input 'x'.
#     '''
#     x = x.float()
#     out, _ = self.lstm(x)
#     out = out[:, -1]  # Get the last output from LSTM sequence
#     out = self.fc(out)  # Pass through the linear layer
#     return out

class CryptoPredictorModel(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers):
    '''
    pass "input_size" as ds.features.shape[2]
    '''
    super(CryptoPredictorModel, self).__init__()
    self.output_size = 1
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, self.output_size)

  def forward(self, x):
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

    out, _ = self.lstm(x, (h0, c0))
    out = out[:, -1, :]  # Get the last output from LSTM sequence
    out = self.fc(out)

    return out
