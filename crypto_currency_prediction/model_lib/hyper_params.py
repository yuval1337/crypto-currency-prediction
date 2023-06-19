import torch

from .typing import *


class HyperParams:
  loss: LossFunction
  optimizer: Optimizer
  lr: float
  epochs: int
  batch_size: int

  def __init__(self,
               epochs: int,
               batch_size: int,
               loss: Literal['nll', 'mse', 'ce'],
               optimizer: Literal['adagrad', 'adam', 'sgd'],
               lr: float) -> None:
    if epochs <= 0:
      raise ValueError
    self.epochs = epochs

    if batch_size <= 0:
      raise ValueError
    self.batch_size = batch_size

    if loss == 'nll':
      self.loss = torch.nn.NLLLoss()
    elif loss == 'mse':
      self.loss = torch.nn.MSELoss(size_average=False, reduce=True)
    else:  # loss == 'ce'
      self.loss = torch.nn.CrossEntropyLoss()

    if optimizer == 'adagrad':
      self.optimizer = torch.optim.Adagrad
    elif optimizer == 'adam':
      self.optimizer = torch.optim.Adam
    else:  # optim == 'sgd'
      self.optimizer = torch.optim.SGD

    if lr >= 1 or lr <= 0:
      raise ValueError
    else:
      self.lr = lr
