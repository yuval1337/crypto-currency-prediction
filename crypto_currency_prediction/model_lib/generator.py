import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from glob import glob

from utils import CryptoCompareConnector as Connector, timestamp, globber
from .dataset import CryptoCompareDataset as Dataset
from .lib_typing import Tuple


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

  def split(self, train: int, valid: int, test: int) -> Tuple[Dataset, Dataset, Dataset]:
    '''Split current dataset into training, validation and testing datasets.'''
    n = len(self.ds.y)
    i = int(np.ceil(n * (train / 100)))
    j = i + int(np.ceil(n * (valid / 100)))

    x_train, x_valid, x_test = np.split(self.ds.x, [i, j])
    y_train, y_valid, y_test = np.split(self.ds.y, [i, j])

    return (
        Dataset(self.ds.symbol, self.ds.to, x_train, y_train),
        Dataset(self.ds.symbol, self.ds.to, x_valid, y_valid),
        Dataset(self.ds.symbol, self.ds.to, x_test, y_test)
    )

  def to_df(self) -> pd.DataFrame:
    data = {key: val
            for key, val in zip(self.FEATURES, zip(*self.ds.x))}
    df = pd.DataFrame(data)
    return df

  def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset

    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    x, y = [], []
    for i in range(len(dataset) - lookback):
      feature = dataset[i:i + lookback]
      target = dataset[i + 1:i + lookback + 1]
      x.append(feature)
      y.append(target)
