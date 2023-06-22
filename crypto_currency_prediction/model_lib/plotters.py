import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .generator import CryptoCompareDatasetGenerator as Generator
from .typing import *


class CryptoDatasetPlotter:
  gen: Generator

  def __init__(self, gen: Generator):
    self.gen = gen

  def show(self) -> None:
    df = self.gen.to_df()
    df.plot(y='close')
    plt.ylabel(f'value ({self.gen.ds.to})')
    plt.xlabel(f'close (per-day)')
    plt.show()

  def show_diff(self, pred: Any) -> None:
    self_df = self.gen.to_df()

    df = pd.DataFrame({
        'label': self_df['close'],
        'pred': pred
    })

    self_df.plot(y=['label', 'pred'], color=['blue', 'red'])

    plt.ylabel(f'value ({self.gen.ds.to})')
    plt.xlabel(f'close (per-day)')
    plt.show()


class TrainingPlotter:
  train_loss: np.ndarray
  hp: HyperParams

  def __init__(self):
    pass
