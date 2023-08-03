import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .dataset import CryptoCompareDataset as Dataset
from .model import CryptoPredictorModel as Model
from .typing import *


class Trainer:
  class _Session:
    '''Bundles training session data regarding training and validation.'''
    model: Model
    train_set: Dataset
    train_dl: DataLoader
    valid_set: Dataset
    valid_dl: DataLoader
    hp: HyperParams
    t_loss: list[float]
    v_loss: list[float]

    def __init__(self,
                 model: Model,
                 hp: HyperParams,
                 train_set: Dataset,
                 valid_set: Dataset) -> None:
      self.model = model
      self.hp = hp
      self.train_set = train_set
      self.valid_set = valid_set
      self.train_dl = train_set.to_dl(hp.batch_size)
      self.valid_dl = valid_set.to_dl(hp.batch_size)
      self.t_loss = []
      self.v_loss = []

  init_model = Model  # initial model before any training were performed
  memory: list[_Session]  # all training sessions performed on this instance

  def __init__(self, model: Model):
    self.init_model = model
    self.memory = []

  def run_train_session(self,
                        train_set: Dataset,
                        valid_set: Dataset,
                        hp: HyperParams,
                        verbose: bool = False) -> None:
    '''Perform a single training session of this object's most recent model, and save the resulting trained model.'''
    line = '{0:<8}{1:<24}{2:<24}'

    session = self._Session(self.model, hp, train_set, valid_set)

    if verbose:
      print('training...')
      print(line.format('epoch', 't_loss', 'v_loss'))

    for epoch in range(session.hp.epochs):
      self._train(session)
      self._validate(session)
      if verbose:
        print(line.format((epoch + 1), session.t_loss[-1], session.v_loss[-1]))

      if hp.has_stopper:
        if hp.stopper(loss=session.t_loss[-1]):  # early stopping triggered
          if verbose:
            print('early-stop triggered!')
          break

    self.memory.append(session)
    if verbose:
      self.plot(session)

  def _train(self, session: _Session) -> None:
    loss = 0.0
    optimizer = session.hp.get_optimizer(session.model.parameters())
    session.model.train()
    for inputs, labels in session.train_dl:
      optimizer.zero_grad()
      outputs = session.model.forward(inputs)
      loss_tensor = session.hp.mse_loss(outputs, labels)
      loss_tensor.backward()
      optimizer.step()
      loss += loss_tensor.item()
    session.t_loss.append(loss / len(session.train_dl))

  def _validate(self, session: _Session) -> float:
    loss = 0.0
    session.model.eval()
    with torch.no_grad():
      for inputs, labels in session.valid_dl:
        outputs = session.model.forward(inputs)
        loss_tensor = session.hp.mse_loss(outputs, labels)
        loss += loss_tensor.item()
    session.v_loss.append(loss / len(session.valid_dl))

  @property
  def model(self) -> Model:
    if len(self.memory) == 0:
      return self.init_model
    else:
      return self.memory[-1].model

  @staticmethod
  def plot(session: _Session) -> None:
    df = pd.DataFrame({'train': session.t_loss, 'valid': session.v_loss})
    df.plot(y=['train', 'valid'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch #')
    plt.show()
