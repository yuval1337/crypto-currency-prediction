from .dataset import CryptoCompareDataset

from .typing import *


class Trainer:
  @staticmethod
  def train(model: Module, hp: HyperParams, df: DataFrame) -> Module:
    '''Performs model training, using given hyper-parameters and data.

    This will use a PyTorch DataLoader that will be created based off the given DataFrame:

      ```
                  time      high       low      open  volumefrom      volumeto     close conversionType conversionSymbol
      0     1557360000   6194.91   5988.65   5998.71    51005.27  3.103366e+08   6171.96         direct
      1     1557446400   6444.57   6133.21   6171.96    68160.13  4.304034e+08   6358.29         direct
      2     1557532800   7394.94   6357.02   6358.29   135248.42  9.331413e+08   7191.36         direct
      3     1557619200   7541.61   6793.21   7191.36   142892.25  1.023203e+09   6977.63         direct
      4     1557705600   8127.75   6873.42   6977.63   149082.57  1.140303e+09   7806.36         direct
      ...          ...       ...       ...       ...         ...           ...       ...            ...              ...
      1496  1686614400  26426.14  25714.90  25905.80    23043.36  5.993846e+08  25925.20         direct
      1497  1686700800  26073.49  24834.48  25925.20    26097.60  6.673818e+08  25126.91         direct
      1498  1686787200  25743.71  24762.12  25126.91    32797.00  8.240060e+08  25575.24         direct
      1499  1686873600  26478.23  25164.41  25575.24    27490.08  7.113611e+08  26329.75         direct
      1500  1686960000  26776.83  26170.63  26329.75    11501.78  3.047019e+08  26511.73         direct

      [1501 rows x 9 columns]
      ```

    Args:
      model (Module): Some deep learning model.
      hp (HyperParams): Bundle of hyper-parameters: loss function, optimizer, learning rate, and total epochs.
      df (DataFrame): A pandas DataFrame consisting of 1500 rows, using the schema mentioned above.

    Returns:
      Module: The trained model.
    '''
    ...
