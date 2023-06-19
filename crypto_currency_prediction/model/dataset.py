from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from .typing import *
from utils import CryptoCompareConnector


class CryptoCompareDataset:
  TIME = 'time'
  FEATURE_LIST = ['high', 'low', 'open', 'volumefrom', 'volumeto', 'close']
  TARGET = 'close'
  AS = 'usd'
  time: np.ndarray
  features: np.ndarray
  target: np.ndarray

  def __init__(self, symbol: str):
    if symbol not in SYMBOLS:
      raise ValueError
    self.symbol = symbol

  def _set_attr_from_df(self, df: DataFrame):
    self.time = df[self.TIME].to_numpy()
    self.features = df[self.FEATURE_LIST].to_numpy()
    self.target = df[self.TARGET].to_numpy()

  def read(self, path: str) -> None:
    df = pd.read_csv(path)
    self._set_attr_from_df(df)

  def fetch(self) -> None:
    df = CryptoCompareConnector.data_histoday(self.symbol)
    self._set_attr_from_df(df)

  def split(self, size: float = 0.2) -> list:
    return train_test_split(self.features, self.target, test_size=0.2)

  def __repr__(self) -> str:
    return 'CryptoDataset(' + f'\nsymbol={self.symbol}, as={self.AS}' + f'\ntime={self.time}' + \
        f'\nfeatures={self.features}' + f'\ntarget={self.target}' + '\n)'
