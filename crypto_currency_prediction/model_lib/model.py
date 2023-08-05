import torch

from utils import timestamp, globber
from .lib_typing import Tensor
from .dataset import CryptoCompareDataset as Dataset


class CryptoPredictorModel(torch.nn.Module):
  '''Simple, hybrid deep-learning model for predicting time-series.  
  Source: https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/
  '''
  in_features: int
  out_features: int
  HIDDEN_SIZE: int = 128
  NUM_LAYERS: int = 2

  def __init__(self, ds: Dataset) -> None:
    super(CryptoPredictorModel, self).__init__()
    self.in_features = ds.x_size
    self.out_features = ds.y_size
    # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    self.lstm = torch.nn.LSTM(self.in_features,
                              self.HIDDEN_SIZE,
                              self.NUM_LAYERS,
                              batch_first=True)
    self.linear = torch.nn.Linear(in_features=self.HIDDEN_SIZE,
                                  out_features=self.out_features)

  def forward(self, x: Tensor) -> Tensor:
    h0 = torch.zeros(self.NUM_LAYERS, x.size(0), self.HIDDEN_SIZE).to(x.device)
    c0 = torch.zeros(self.NUM_LAYERS, x.size(0), self.HIDDEN_SIZE).to(x.device)
    out, _ = self.lstm(x, (h0, c0))
    out = out[:, -1, :]  # Get the last output from LSTM sequence
    out = self.linear(out)
    return out

  def predict(self, x: Tensor) -> Tensor:
    self.eval()
    with torch.no_grad():
      out = self.forward(x)
    return out

  def save(self) -> None:
    '''Saves this model to a local `.pth` file.'''
    torch.save(
        obj=self.state_dict(),
        f=f'cpm_{timestamp()}.pth'
    )

  # TODO finish this method
  def load(self) -> None:
    '''Loads this model with values from the most recent local `.pth` file.'''
    # file = globber(f'cpm_*.pth')
    pass
