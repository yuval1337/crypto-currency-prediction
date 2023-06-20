import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset


from .dataset import CryptoCompareDataset as Dataset
from .model import CryptoPredictorModel as Model
from .typing import *


def test(model_arg: Model, hp: HyperParams, ds: Dataset) -> pd.DataFrame:
  tds = TensorDataset(ds.get_x, ds.get_y)
  dl = DataLoader(tds, hp.batch_size, shuffle=False)
  model = model_arg

  print('testing...')
  model.eval()
  predictions = []
  with torch.no_grad():
    test_loss = 0.0
    for inputs, labels in dl:
      pred = model.forward(inputs)
      predictions.append(pred)
      loss = hp.calc_loss(pred, labels)
      test_loss += (loss.item() / len(inputs))

  print(f'        test_loss={test_loss}')

  return pd.DataFrame({'close': predictions})
