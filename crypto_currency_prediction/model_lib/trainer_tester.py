import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


from .dataset import CryptoCompareDataset as Dataset
from .model import CryptoPredictorModel as Model
from .typing import *


def train(model_arg: Model, hp: HyperParams, ds: Dataset) -> Model:
  tds = TensorDataset(ds.get_X, ds.get_y)
  dl = DataLoader(tds, hp.batch_size, shuffle=True)
  model = model_arg
  model.train()
  optimizer = hp.optimizer(model.parameters(), lr=hp.lr)
  for epoch in range(hp.epochs):
    epoch_loss = 0.0
    for inputs, labels in dl:
      optimizer.zero_grad()
      outputs = model.forward(inputs)
      loss = hp.loss(outputs, labels)
      loss.backward()
      optimizer.step()
      epoch_loss += (loss / hp.batch_size)
    print(f'epoch={epoch + 1}, loss={epoch_loss}')
  return model


def test(model_arg: Model, hp: HyperParams, ds: Dataset):
  tds = TensorDataset(ds.get_X, ds.get_y)
  dl = DataLoader(tds, hp.batch_size, shuffle=True)
  model = model_arg
  model.eval()

  # with torch.no_grad():
  #   pred = model.forward(ds.features_as_tensor).squeeze().numpy()

  # pred_denorm = pred * np.std(ds.target_as_tensor) + \
  #     np.mean(ds.target_as_tensor)

  # print(pred_denorm)
