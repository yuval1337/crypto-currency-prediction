import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset


from .dataset import CryptoCompareDataset as Dataset
from .model import CryptoPredictorModel as Model
from .typing import *


def test(model_arg: Model, hp: HyperParams, ds: Dataset) -> pd.DataFrame:
  tds = TensorDataset(ds.x_scaled, ds.y_scaled)
  dl = DataLoader(tds,
                  hp.batch_size,
                  shuffle=False,
                  num_workers=0
                  )
  model = model_arg

  print('testing...')
  model.eval()
  predictions = []
  with torch.no_grad():
    test_loss = 0.0
    for inputs, labels in dl:  # batch
      batch_prediction = model.forward(inputs)
      predictions.append(ds.descale_tensor(batch_prediction).flatten())
      loss = hp.calc_loss(batch_prediction, labels)
      test_loss += (loss.item() / len(inputs))

  predictions = np.concatenate(predictions)

  print(f'        test_loss={test_loss}')

  return pd.DataFrame({'close': predictions})
