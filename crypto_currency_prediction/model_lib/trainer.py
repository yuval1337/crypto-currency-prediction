import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from .dataset import CryptoCompareDataset as Dataset
from .model import CryptoPredictorModel as Model
from .typing import *


class TrainingSession:
  '''Bundle a single training session's results.'''
  model: Model
  train_set: Dataset
  hp: HyperParams
  losses: list[float]
  early_stop: bool  # whether an early-stop was triggered

  def __init__(self,
               model: Model,
               train_set: Dataset,
               hp: HyperParams,
               losses: list[float]) -> None:
    self.model = model
    self.train_set = train_set
    self.hp = hp
    self.losses = losses
    self.early_stop = True if len(losses) < hp.epochs else False


class Trainer:
  init_model = Model  # initial model before any training were performed
  # all training sessions performed using this instance
  memory: list[TrainingSession]

  def __init__(self, model: Model):
    self.init_model = model
    self.memory = []

  def train(self,
            train_set: Dataset,
            hp: HyperParams = None,
            verbose: bool = True) -> TrainingSession | None:
    '''Perform a single training session of this object's most recent model, and save the resulting trained model'''
    if verbose:
      print('training...')

    model = self.model  # get the most recent model
    optimizer = hp.get_optimizer(model.parameters())
    dl = train_set.to_dl(hp.batch_size)

    losses = []

    model.train()
    for epoch in range(hp.epochs):
      epoch_loss = 0.0
      for inputs, labels in dl:
        optimizer.zero_grad()
        outputs = model.forward(inputs)
        loss = hp.mse_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += (loss.item() / len(inputs))
      losses.append(epoch_loss)
      if verbose:
        print(f'epoch={epoch + 1}, loss={epoch_loss}')

      if hp.has_stopper:
        if hp.stopper(loss=epoch_loss):  # early stopping triggered
          if verbose:
            print('early-stop triggered!')
          break

    session = TrainingSession(
        model,
        train_set,
        hp,
        losses
    )

    self.memory.append(session)

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
        loss = self.hp.mse_loss(batch_prediction, labels)
        test_loss += (loss.item() / len(inputs))

    predictions = np.concatenate(predictions)

    if verbose:
      print(f'test_loss={test_loss}')
      print(predictions)

    # df = pd.DataFrame({'close': predictions})
    return None

  @property
  def model(self) -> Model:
    if len(self.memory) == 0:
      return self.init_model
    else:
      return self.memory[-1].model
