import torch
import numpy as np

from .typing import *


class CryptoCompareDataset:
  X: np.ndarray  # features
  y: np.ndarray  # target
  symbol: str

  def __init__(self, symbol: str, X: np.ndarray, y: np.ndarray):
    self.symbol = symbol
    self.X = X
    self.y = y

  @property
  def get_X(self) -> Tensor:
    normalized = (self.X - np.mean(self.X, axis=0)
                  ) / np.std(self.X, axis=0)
    as_tensor = torch.from_numpy(normalized)
    return as_tensor.view(as_tensor.shape[0], 1, -1).float()

  @property
  def get_y(self) -> Tensor:
    normalized = (self.y - np.mean(self.y)
                  ) / np.std(self.y)
    as_tensor = torch.from_numpy(normalized).float()
    return as_tensor

  def __repr__(self) -> str:
    return 'CryptoDataset(' + f'\nsymbol={self.symbol}, as={self.AS}' + f'\ntime={self._time}' + \
        f'\nfeatures={self._features}' + f'\ntarget={self._target}' + '\n)'
