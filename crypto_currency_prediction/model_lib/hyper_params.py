from .typing import *
from torch.nn.functional import mse_loss


class HyperParams:
  def __init__(self,
               epochs: int,
               batch_size: int,
               lr: float,
               loss_func: Any,
               optimizer: Optimizer,
               reg_factor: float = 0.0) -> None:
    self.epochs = epochs
    self.batch_size = batch_size
    self.loss_func = loss_func
    self.lr = lr
    self.optimizer = optimizer
    self.reg_factor = reg_factor

  def get_optimizer(self, model_params) -> Optimizer:
    return self.optimizer(
        model_params,
        lr=self.lr,
        weight_decay=self.reg_factor
    )

  # TODO currently supports MSE only
  def calc_loss(self, pred, target) -> Tensor:
    return self.loss_func(pred.view(-1), target, reduction='mean')
