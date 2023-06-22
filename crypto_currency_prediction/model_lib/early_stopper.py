from .typing import *


class EarlyStopper:
  '''Implementation of an early-stopping mechanism for model training.'''
  patience: int  # number of epochs to wait before stopping
  delta: float  # minimum improvement required to be considered as an improvement
  best_loss: float
  counter: int

  def __init__(self, tolerance: Literal['low', 'med', 'high'] = 'low'):
    if tolerance == 'low':
      self.patience, self.delta = (3, 0.1)
    elif tolerance == 'med':
      self.patience, self.delta = (5, 0.01)
    else:  # tolerance == 'high'
      self.patience, self.delta = (10, 0.001)
    self.best_loss = float('inf')
    self.counter = 0

  def __call__(self, loss: float) -> bool:
    if loss < (self.best_loss - self.delta):
      self.best_loss = loss
      self.counter = 0
    else:
      self.counter += 1
      if self.counter == self.patience:
        return True  # trigger early stop
    return False
