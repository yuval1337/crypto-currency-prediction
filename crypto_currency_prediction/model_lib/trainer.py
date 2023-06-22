import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .dataset import CryptoCompareDataset as Dataset
from .model import CryptoPredictorModel as Model
from .typing import *


LINE = '{0:<8}{1:<20}{2:<20}'


class TrainingSession:
  '''Bundle a single training session's results.'''
  model: Model
  train_set: Dataset
  test_set: Dataset
  hp: HyperParams
  train_loss_list: list[float]
  test_loss_list: list[float]

  def __init__(self,
               model: Model,
               train_set: Dataset,
               test_set: Dataset,
               hp: HyperParams,
               train_loss_list: list[float],
               test_loss_list: list[float]) -> None:
    self.model = model
    self.train_set = train_set
    self.test_set = test_set
    self.hp = hp
    self.train_loss_list = train_loss_list
    self.test_loss_list = test_loss_list


class Trainer:
  init_model = Model  # initial model before any training were performed
  # all training sessions performed using this instance
  memory: list[TrainingSession]

  def __init__(self, model: Model):
    self.init_model = model
    self.memory = []

  def train(self,
            train_set: Dataset,
            test_set: Dataset,
            hp: HyperParams,
            verbose: bool = False) -> TrainingSession | None:
    '''Perform a single training session of this object's most recent model, and save the resulting trained model'''
    if verbose:
      print('training...')
    model = self.model  # get the most recent model
    optimizer = hp.get_optimizer(model.parameters())
    train_dl = train_set.to_dl(hp.batch_size)
    test_dl = test_set.to_dl(hp.batch_size)

    train_loss_list = []
    test_loss_list = []

    if verbose:
      print(LINE.format('epoch', 'train_loss', 'test_loss'))

    for epoch in range(hp.epochs):
      train_loss = 0.0
      test_loss = 0.0
      model.train()
      for inputs, labels in train_dl:
        optimizer.zero_grad()
        outputs = model.forward(inputs)
        loss = hp.mse_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += (loss.item() / len(inputs))
      model.eval()
      with torch.no_grad():
        for inputs, labels in test_dl:
          outputs = model.forward(inputs)
          loss = hp.mse_loss(outputs, labels)
          test_loss += (loss.item() / len(inputs))
      if verbose:
        print(LINE.format((epoch + 1), train_loss, test_loss))
      if hp.has_stopper:
        if hp.stopper(loss=train_loss):  # early stopping triggered
          if verbose:
            print('early-stop triggered!')
          break
      train_loss_list.append(train_loss)
      test_loss_list.append(test_loss)

    session = TrainingSession(
        model,
        train_set,
        test_set,
        hp,
        train_loss_list,
        test_loss_list
    )
    if verbose:
      self.plot(session)
    self.memory.append(session)

  @staticmethod
  def plot(ts: TrainingSession) -> None:
    df = pd.DataFrame({'train': ts.train_loss_list,
                      'test': ts.test_loss_list})
    df.plot(y=['train', 'test'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch #')
    plt.show()

  @property
  def model(self) -> Model:
    if len(self.memory) == 0:
      return self.init_model
    else:
      return self.memory[-1].model
