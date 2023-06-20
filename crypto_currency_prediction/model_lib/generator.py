from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from datetime import datetime
from glob import glob

from .typing import *
from .dataset import CryptoCompareDataset as Dataset
from utils import CryptoCompareConnector as Connector


class CryptoCompareDatasetGenerator:
  FEATURES = ['high', 'low', 'open', 'volumefrom', 'volumeto', 'close']
  TARGET = 'close'

  ds: Dataset

  def __init__(self, symbol: str, to: str, from_file: bool = False):
    if from_file:
      csv_files = glob(f'{symbol}2{to}_*.csv')
      if csv_files == []:
        raise RuntimeError
      self.read(symbol, to, csv_files[-1])  # use the latest csv
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
    timestamp = datetime.now().strftime('%d%m%y%H%M%S')
    df.to_csv(symbol + '2' + to + '_' + timestamp + '.csv')

    x, y = self._df_to_x_y(df)
    self.ds = Dataset(symbol, to, x, y)

  def read(self, symbol: str, to: str, path: str) -> None:
    '''Read a dataset from a .csv file and set it as this object's `ds`.'''
    df = pd.read_csv(path)
    x, y = self._df_to_x_y(df)
    self.ds = Dataset(symbol, to, x, y)

  def split(self, train_size: float) -> Tuple[Dataset, Dataset]:
    '''Split current (state) dataset into train test datasets, and return them.'''
    x_train, x_test, y_train, y_test = train_test_split(
        self.ds.x,
        self.ds.y,
        train_size=train_size,
        test_size=(1.0 - train_size)
    )
    return (
        Dataset(self.ds.symbol, self.ds.to, x_train, y_train),
        Dataset(self.ds.symbol, self.ds.to, x_test, y_test)
    )

  def to_df(self) -> pd.DataFrame:
    data = {key: val
            for key, val in zip(self.FEATURES, zip(*self.ds.x))}
    df = pd.DataFrame(data)
    return df
