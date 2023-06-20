import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


from .dataset import CryptoCompareDataset as Dataset
from .model import CryptoPredictorModel as Model
from .typing import *


def train(model_arg: Model, hp: HyperParams, ds: Dataset) -> Model:
  tds = TensorDataset(ds.get_x, ds.get_y)
  dl = DataLoader(tds, hp.batch_size, shuffle=True)
  model = model_arg
  optimizer = hp.get_optimizer(model.parameters())

  print('training...')
  model.train()
  for epoch in range(hp.epochs):
    train_loss = 0.0
    for inputs, labels in dl:
      optimizer.zero_grad()
      outputs = model.forward(inputs)
      loss = hp.calc_loss(outputs, labels)
      loss.backward()
      optimizer.step()
      train_loss += (loss.item() / len(inputs))
    print(f'epoch={epoch + 1}, train_loss={train_loss}')

  return model
