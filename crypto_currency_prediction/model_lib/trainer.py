import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from .dataset import CryptoCompareDataset as Dataset
from .model import CryptoPredictorModel as Model
from .typing import *


class EarlyStopper:
  def __init__(self, patience: int, delta: float):
    self.patience = patience  # Number of epochs to wait before stopping
    self.delta = delta  # Minimum improvement required to be considered as an improvement
    self.best_loss = float('inf')
    self.counter = 0

  def check_stop(self, current_loss):
    if current_loss < self.best_loss - self.delta:
      self.best_loss = current_loss
      self.counter = 0
    else:
      self.counter += 1
      if self.counter >= self.patience:
        print('EarlyStopper triggered!')
        return True  # Stop training
    return False  # Continue training


class Trainer:
  models: list[Model]
  hp: HyperParams
  ds: Dataset

  def __init__(self, model: Model, hp: HyperParams, ds: Dataset):
    self.models = [model]
    self.hp = hp
    self.ds = ds

  def train(self) -> None:
    '''Perform a single training session of this object's latest model, save the trained model'''
    tds = TensorDataset(self.ds.x_scaled, self.ds.y_scaled)
    dl = DataLoader(tds, self.hp.batch_size, shuffle=True)
    model = self.models[-1]  # get the latest model
    optimizer = self.hp.get_optimizer(model.parameters())
    es = EarlyStopper(patience=5, delta=0.001)

    print('training...')
    model.train()
    for epoch in range(self.hp.epochs):
      epoch_loss = 0.0
      for inputs, labels in dl:
        optimizer.zero_grad()
        outputs = model.forward(inputs)
        loss = self.hp.calc_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += (loss.item() / len(inputs))

      print(f'epoch={epoch + 1}, loss={epoch_loss}')
      # Check for early stopping
      if es.check_stop(epoch_loss):
        break

    self.models.append(model)

  @property
  def latest_model(self) -> Model:
    return self.models[-1]
