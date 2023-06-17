import torch
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

# TODO more work is due here..


class DataloaderGenerator:
  @staticmethod
  def gen(data: pd.DataFrame, batch_size: int = 32) -> DataLoader:
    # Assuming the DataFrame has the following columns: ['time', 'high', 'low', 'open', 'volumefrom', 'volumeto', 'close']
    # Extract the input features and target variable
    features = data[['time', 'high', 'low', 'open',
                     'volumefrom', 'volumeto', 'close']].values
    target = data['close'].values

    # Convert to PyTorch tensors
    features_tensor = torch.Tensor(features)
    target_tensor = torch.Tensor(target)

    # Create a TensorDataset
    dataset = TensorDataset(features_tensor, target_tensor)

    return DataLoader(dataset, batch_size, shuffle=True)
