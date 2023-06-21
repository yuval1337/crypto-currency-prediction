import torch
import torch.nn as nn
from .typing import *

# https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/


class CryptoPredictorModel(nn.Module):
  '''Simple, hybrid deep-learning model for predicting time-series.'''

  def __init__(self, in_features, out_features, hidden_size, num_layers) -> None:
    super(CryptoPredictorModel, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    self.lstm = nn.LSTM(in_features, hidden_size, num_layers, batch_first=True)
    # in size must be hidden size!
    self.linear = nn.Linear(hidden_size, out_features)

  def forward(self, x: Tensor) -> Tensor:
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
    out, _ = self.lstm(x, (h0, c0))
    out = out[:, -1, :]  # Get the last output from LSTM sequence
    out = self.linear(out)
    return out
