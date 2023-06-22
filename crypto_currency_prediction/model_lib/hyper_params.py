from .typing import *
import torch.nn.functional as F
from .early_stopper import EarlyStopper


class HyperParams:
  '''Bundle model training hyper-parameters.'''
  epochs: int
  batch_size: int
  _learn_rate: float
  optimizer: Optimizer
  _reg_factor: float
  stopper: EarlyStopper

  def __init__(self,
               epochs: int,
               batch_size: int,
               optimizer: Optimizer,
               learn_rate: float = None,
               reg_factor: float = None,
               stopper: EarlyStopper = None) -> None:
    self.epochs = epochs
    self.batch_size = batch_size
    self.optimizer = optimizer

    self._learn_rate = learn_rate
    self._reg_factor = reg_factor
    self.stopper = stopper

  def get_optimizer(self, model_params) -> Optimizer:
    if self._learn_rate is None:
      if self._reg_factor is None:
        return self.optimizer(model_params)
      else:  # self._reg_factor is not None:
        return self.optimizer(model_params, weight_decay=self._reg_factor)
    else:  # self._learn_rate is not None
      if self._reg_factor is None:
        return self.optimizer(model_params, lr=self._learn_rate)
      else:  # self._reg_factor is not None:
        return self.optimizer(model_params, lr=self._learn_rate, weight_decay=self._reg_factor)

  def mse_loss(self, pred: Tensor, target: Tensor) -> Tensor:
    return F.mse_loss(pred, target, reduction='mean')

  @property
  def has_stopper(self) -> bool:
    return False if (self.stopper is None) else True
