import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .dataset import CryptoCompareDataset as Dataset
from .model import CryptoPredictorModel as Model
from .hyper_params import Hyperparams
from .lib_typing import DataLoader, Tensor, Optimizer

import matplotlib.dates as mdates
from datetime import datetime, timedelta


class Trainer:
  class _Session:
    '''Bundles single session's data, regarding training and validation.'''
    model: Model
    hp: Hyperparams
    train_ds: Dataset
    valid_ds: Dataset
    train_loss: list[float]
    valid_loss: list[float]

    def __init__(self,
                 model: Model,
                 hp: Hyperparams,
                 train_ds: Dataset,
                 valid_ds: Dataset) -> None:
      self.model = model
      self.hp = hp
      self.train_ds = train_ds
      self.valid_ds = valid_ds
      self.train_loss = []
      self.valid_loss = []

    def plot(self):
      df = pd.DataFrame({'Training': self.train_loss,
                        'Validation': self.valid_loss})
      df.plot(y=['Training', 'Validation'])
      plt.ylabel('Loss')
      plt.xlabel('Epoch')
      plt.show()

  init_model = Model  # initial model before any training were performed
  memory: list[_Session]  # all training sessions performed on this instance

  def __init__(self, model: Model):
    self.init_model = model
    self.memory = []

  def run_train_session(self,
                        train_ds: Dataset,
                        valid_ds: Dataset,
                        hp: Hyperparams,
                        verbose: bool = False,
                        plot: bool = False) -> None:
    '''Perform a single training session of this object's most recent model, and save the resulting trained model.'''

    session = self._Session(self.model, hp, train_ds, valid_ds)

    for epoch in range(session.hp.epochs):
      self._train(session)
      self._validate(session)

      if verbose:
        line = '{0:<8}{1:<24}{2:<24}'
        if epoch == 0:
          print(line.format('epoch', 'training loss', 'validation loss'))
        print(line.format((epoch + 1),
              session.train_loss[-1], session.valid_loss[-1]))

      if hp.has_stopper:
        if hp.stopper(loss=session.train_loss[-1]):  # early stopping triggered
          if verbose:
            print('early stopper triggered!')
          break

    self.memory.append(session)
    if plot:
      session.plot()

  def _train(self, session: _Session) -> None:
    loss_per_batch = []
    optimizer_obj: Optimizer = session.hp.optimizer(session.model.parameters())
    dl: DataLoader = session.train_ds.to_dl(session.hp.batch_size)

    session.model.train()
    for inputs, labels in dl:
      optimizer_obj.zero_grad()
      outputs = session.model.forward(inputs)
      loss: Tensor = session.hp.loss_fn(input=outputs, target=labels)
      loss.backward()
      optimizer_obj.step()
      loss_per_batch.append(loss.item())

    session.train_loss.append(np.mean(loss_per_batch))

  def _validate(self, session: _Session) -> float:
    loss_per_batch = []
    dl: DataLoader = session.valid_ds.to_dl(session.hp.batch_size)

    session.model.eval()
    with torch.no_grad():
      for inputs, labels in dl:
        outputs = session.model.forward(inputs)
        loss: Tensor = session.hp.loss_fn(input=outputs, target=labels)
        loss_per_batch.append(loss.item())

    session.valid_loss.append(np.mean(loss_per_batch))

  def run_test(self, test_ds: Dataset) -> None:
    dl = test_ds.to_dl(1)

    self.model.eval()
    with torch.no_grad():
      actual_values = []
      predicted_values = []

      for inputs, labels in dl:
        predicted_scaled = self.model.predict(inputs)
        actual_scaled = labels  # Assuming labels are already in the original scale

        predicted = test_ds.descale_y(predicted_scaled)
        actual = test_ds.descale_y(actual_scaled)

        actual_values.append(actual.item())
        predicted_values.append(predicted.item())

      plt.figure(figsize=(10, 6))
      plt.plot(actual_values, label=f'Actual Price')
      plt.plot(predicted_values, label=f'Predicted Price')
      plt.xlabel('Day')
      plt.ylabel(f'Price ({test_ds.to.upper()})')
      plt.title(f'{test_ds.symbol.upper()} to {test_ds.to.upper()}')

      plt.legend()
      plt.tight_layout()
      plt.show()

  @property
  def model(self) -> Model:
    if len(self.memory) == 0:
      return self.init_model
    else:
      return self.memory[-1].model
