import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .typing import *


class Feature(np.ndarray):
  pass


class Target(np.ndarray):
  pass


class CryptoCompareDataset:
  symbol: str  # type of cryptocurrency: BTC, ETH, etc.
  to: str  # valuation currency: USD, EUR, etc.
  x: np.ndarray  # features
  y: np.ndarray  # targets
  type: Literal['neutral', 'train', 'test']
  scaler: MinMaxScaler

  def __init__(self,
               symbol: str,
               to: str,
               x: np.ndarray,
               y: np.ndarray,
               type: Literal['neutral', 'train', 'test'] = 'neutral') -> None:
    self.symbol = symbol
    self.to = to
    self.x = x
    self.y = y
    self.type = type
    self.scaler = MinMaxScaler()

  @property
  def x_size(self) -> int:
    return 1 if len(self.x.shape) == 1 else self.x.shape[1]

  @property
  def y_size(self) -> int:
    return 1 if len(self.y.shape) == 1 else self.y.shape[1]

  @property
  def x_scaled(self) -> Tensor:
    x_scaled = self.scaler.fit_transform(self.x, self.y)
    x_scaled_as_tensor = self.np2tensor(x_scaled).float()
    return x_scaled_as_tensor

  @property
  def y_scaled(self) -> Tensor:
    y_scaled = self.scaler.fit_transform(self.y.reshape(-1, 1))
    y_scaled_as_tensor = self.np2tensor(y_scaled).float()
    return y_scaled_as_tensor

  def to_dl(self, batch_size: int) -> DataLoader:
    tds = TensorDataset(self.x_scaled, self.y_scaled.squeeze(2))
    dl = DataLoader(tds,
                    batch_size,
                    shuffle=(True if (self.type == 'train') else False))
    return dl

  def scale_x(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return self.scaler.fit_transform(x, y)

  def scale_y(self, y: np.ndarray) -> np.ndarray:
    return self.scaler.fit_transform(y.reshape(-1, 1))

  def descale_tensor(self, tensor: Tensor) -> np.ndarray:
    arr = tensor.numpy()
    return self.scaler.inverse_transform(arr)

  @ staticmethod
  def np2tensor(x: np.ndarray) -> Tensor:
    if len(x.shape) > 1:
      tensor = torch.from_numpy(x)
      return tensor.view(tensor.shape[0], 1, -1)
    else:
      return torch.from_numpy(x)

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
