from .lib_typing import Optimizer, Literal, Tensor, Callable
from .early_stopper import EarlyStopper


class Hyperparams:
  '''Bundles hyperparameters for model training.'''
  loss_fn: Callable
  optimizer: Optimizer
  epochs: int
  batch_size: int
  learn_rate: float
  reg_factor: float
  stopper: EarlyStopper

  def __init__(self,
               loss_fn: Callable,
               optimizer: Optimizer,
               epochs: int,
               batch_size: int,
               learn_rate: float,
               reg_factor: float,
               tolerance: Literal['low', 'med', 'high', 'off']) -> None:
    self.loss_fn = loss_fn
    self.optimizer = optimizer
    self.epochs = epochs
    self.batch_size = batch_size
    self.learn_rate = learn_rate
    self.reg_factor = reg_factor
    if tolerance != 'off':
      self.stopper = EarlyStopper(tolerance)
    else:
      self.stopper = None

  def get_optimizer(self, model_params) -> Optimizer:
    if self.learn_rate == None:
      if self.reg_factor == None:
        return self.optimizer(model_params)
      else:  # reg_factor != None:
        return self.optimizer(model_params, weight_decay=self.reg_factor)
    else:  # learn_rate != None
      if self.reg_factor == None:
        return self.optimizer(model_params, lr=self.learn_rate)
      else:  # reg_factor != None:
        return self.optimizer(model_params, lr=self.learn_rate, weight_decay=self.reg_factor)

  @property
  def has_stopper(self) -> bool:
    return self.stopper != None
