import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

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
  memory: list[Model]
  hp: HyperParams
  train_set: Dataset
  test_set: Dataset
  es: EarlyStopper

  def __init__(self, model: Model,
               hp: HyperParams,
               train_set: Dataset,
               test_set: Dataset):
    self.memory = [model]
    self.hp = hp
    self.train_set = train_set
    self.test_set = test_set
    self.es = EarlyStopper(patience=5, delta=0.001)

  def train(self, verbose: bool = False) -> None:
    '''Perform a single training session of this object's latest model, save the trained model'''
    if verbose:
      print('training...')

    model = self.model
    optimizer = self.hp.get_optimizer(model.parameters())

    model.train()
    for epoch in range(self.hp.epochs):
      epoch_loss = 0.0
      for inputs, labels in self.train_dl:
        optimizer.zero_grad()
        outputs = model.forward(inputs)
        loss = self.hp.calc_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += (loss.item() / len(inputs))
      if verbose:
        print(f'epoch={epoch + 1}, loss={epoch_loss}')
      if self.es.check_stop(epoch_loss):  # early stopping mechanism
        break

    self.memory.append(model)

  def test(self, verbose: bool = True) -> None:
    if verbose:
      print('testing...')

    model = self.model
    predictions = []

    model.eval()
    with torch.no_grad():
      test_loss = 0.0
      for inputs, labels in self.test_dl:
        batch_prediction = model.forward(inputs)
        predictions.append(self.test_set.descale_tensor(
            batch_prediction).flatten())
        loss = self.hp.calc_loss(batch_prediction, labels)
        test_loss += (loss.item() / len(inputs))

    predictions = np.concatenate(predictions)

    if verbose:
      print(f'test_loss={test_loss}')
      print(predictions)

    # df = pd.DataFrame({'close': predictions})
    return None

  @property
  def train_dl(self) -> DataLoader:
    tds = TensorDataset(self.train_set.x_scaled,
                        self.train_set.y_scaled.squeeze(2))
    dl = DataLoader(tds, self.hp.batch_size, shuffle=True)
    return dl

  @property
  def test_dl(self) -> DataLoader:
    tds = TensorDataset(self.test_set.x_scaled,
                        self.test_set.y_scaled.squeeze(2))
    dl = DataLoader(tds, self.hp.batch_size, shuffle=False)
    return dl

  @property
  def model(self) -> Model:
    return self.memory[-1]
