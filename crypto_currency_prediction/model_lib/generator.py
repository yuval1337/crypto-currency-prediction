from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from datetime import datetime


from .typing import *
from .dataset import CryptoCompareDataset as Dataset
from utils import CryptoCompareConnector as Connector


TIME = 'time'
FEATURE_LIST = ['high', 'low', 'open', 'volumefrom', 'volumeto', 'close']
TARGET = 'close'
AS = 'usd'


class CryptoCompareDatasetGenerator:
  ds: Dataset

  def df_to_X_y(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    _ = df[TIME].to_numpy()
    X = df[FEATURE_LIST].to_numpy()
    y = df[TARGET].to_numpy()
    return (X, y)

  def fetch(self, symbol: str) -> None:
    '''Fetch data from CryptoCompare's service (public API), save it as a local pandas DataFrame, and set it as this object's ds.'''
    df = Connector.data_histoday(symbol)
    timestamp = datetime.now().strftime('%d%m%y%H%M%S')
    df.to_csv(symbol + '_' + timestamp + '.csv')
    X, y = self.df_to_X_y(df)
    self.ds = Dataset(symbol, X, y)

  def read(self, symbol: str, path: str) -> None:
    '''Read a dataset from a .csv file and set it as this object's ds'''
    df = pd.read_csv(path)
    X, y = self.df_to_X_y(df)
    self.ds = Dataset(symbol, X, y)

  def split(self, train_size: float) -> Tuple[Dataset, Dataset]:
    '''Split current (state) dataset into train test datasets, and return them.'''
    X_train, X_test, y_train, y_test = train_test_split(
        self.ds.X,
        self.ds.y,
        train_size=train_size,
        test_size=(1 - train_size),
        random_state=42
    )
    return (
        Dataset(self.ds.symbol, X_train, y_train),
        Dataset(self.ds.symbol, X_test, y_test)
    )
