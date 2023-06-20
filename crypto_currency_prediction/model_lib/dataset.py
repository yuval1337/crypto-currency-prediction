import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .typing import *


class CryptoCompareDataset:
  symbol: str  # type of cryptocurrency
  # the currency in which the crypto value is measured, e.g. USD, EUR, ...
  to: str
  x: np.ndarray  # features
  y: np.ndarray  # target
  scaler: MinMaxScaler

  def __init__(self, symbol: str, to: str, x: np.ndarray, y: np.ndarray):
    self.symbol = symbol
    self.to = to
    self.x = x
    self.y = y
    self.scaler = MinMaxScaler()

  @property
  def get_x(self) -> Tensor:
    normalized = self.normalize(self.x)
    return self.np2tensor(normalized).float()

  @property
  def get_y(self) -> Tensor:
    normalized = self.normalize(self.y)
    return self.np2tensor(normalized).float()

  @staticmethod
  def np2tensor(x: np.ndarray) -> Tensor:
    if len(x.shape) > 1:
      tensor = torch.from_numpy(x)
      return tensor.view(tensor.shape[0], 1, -1)
    else:
      return torch.from_numpy(x)

  @staticmethod
  def tensor2np(x: Tensor) -> np.ndarray:
    return x.numpy()

  def normalize(self, x: np.ndarray) -> np.ndarray:
    # return (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    return self.scaler.fit_transform(x.reshape(-1, 1))

  def denormalize(self, x_scaled: np.ndarray) -> np.ndarray:
    # return (x * np.std(x, axis=0)) + np.mean(x, axis=0)
    return self.scaler.inverse_transform(x_scaled.reshape(-1, 1))

  def __repr__(self) -> str:
    return 'CryptoCompareDataset(\n'\
        + f'symbol={self.symbol}\n'\
        + f'to={self.to}\n'\
        + f'features={self.x}\n'\
        + f'target={self.y}\n'\
        + ')'

  def __eq__(self, other) -> bool:
    if not isinstance(other, CryptoCompareDataset):
      raise TypeError
    return all([
        self.symbol == other.symbol,
        self.to == other.to
    ])
