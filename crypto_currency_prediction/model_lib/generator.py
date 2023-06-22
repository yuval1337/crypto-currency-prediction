from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from glob import glob

from .typing import *
from .dataset import CryptoCompareDataset as Dataset
from utils import CryptoCompareConnector as Connector, timestamp, globber


class CryptoCompareDatasetGenerator:
  FEATURES = ['high', 'low', 'open', 'volumefrom', 'volumeto', 'close']
  TARGET = 'close'

  ds: Dataset

  def __init__(self, symbol: str, to: str, from_file: bool = False):
    if from_file:
      # use the most recent local file
      self.read(symbol, to, globber(f'{symbol}2{to}_*.csv'))
    else:
      self.fetch(symbol, to)  # get fresh data from the api

  def _df_to_x_y(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    x = df[self.FEATURES].to_numpy()
    y = df[self.TARGET].to_numpy()
    return (x, y)

  def fetch(self, symbol: str, to: str) -> None:
    '''Fetch data from CryptoCompare's service (public API), save it as a local pandas DataFrame, and set it as this object's `ds`.'''
    df = Connector.data_histoday(symbol, to)

    # save the DataFrame to a local csv file
    df.to_csv(symbol + '2' + to + '_' + timestamp() + '.csv')

    x, y = self._df_to_x_y(df)
    self.ds = Dataset(symbol, to, x, y)

  def read(self, symbol: str, to: str, path: str) -> None:
    '''Read a dataset from a .csv file and set it as this object's `ds`.'''
    df = pd.read_csv(path)
    x, y = self._df_to_x_y(df)
    self.ds = Dataset(symbol, to, x, y)

  def split(self, ratio: float) -> Tuple[Dataset, Dataset]:
    '''Split current (state) dataset into train test datasets, and return them.'''
    idx = int(len(self.ds.y) * ratio)

    x_train, x_test = np.split(self.ds.x, [idx])
    y_train, y_test = np.split(self.ds.y, [idx])

    return (
        Dataset(self.ds.symbol, self.ds.to, x_train, y_train),
        Dataset(self.ds.symbol, self.ds.to, x_test, y_test)
    )

  def to_df(self) -> pd.DataFrame:
    data = {key: val
            for key, val in zip(self.FEATURES, zip(*self.ds.x))}
    df = pd.DataFrame(data)
    return df
