import torch
import numpy as np

from .typing import *


class CryptoCompareDataset:
  symbol: str  # type of cryptocurrency
  # the currency in which the crypto value is measured, e.g. USD, EUR, ...
  to: str
  x: np.ndarray  # features
  y: np.ndarray  # target

  def __init__(self, symbol: str, to: str, x: np.ndarray, y: np.ndarray):
    self.symbol = symbol
    self.to = to
    self.x = x
    self.y = y

  @property
  def get_x(self) -> Tensor:
    normalized = (self.x - np.mean(self.x, axis=0)
                  ) / np.std(self.x, axis=0)
    as_tensor = torch.from_numpy(normalized)
    return as_tensor.view(as_tensor.shape[0], 1, -1).float()

  @property
  def get_y(self) -> Tensor:
    normalized = (self.y - np.mean(self.y)
                  ) / np.std(self.y)
    as_tensor = torch.from_numpy(normalized).float()
    return as_tensor

  def __repr__(self) -> str:
    return 'CryptoCompareDataset(\n'\
        + f'symbol={self.symbol}\n'\
        + f'to={self.to}\n'\
        + f'features={self.x}\n'\
        + f'target={self.y}\n'\
        + ')'
