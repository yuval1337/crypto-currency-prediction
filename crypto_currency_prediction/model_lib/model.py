import torch
import torch.nn as nn
from .typing import *


class CryptoPredictorModel(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers):
    '''
    Pass the argument "input_size" as `ds.features.shape[2]`.
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
