from utils import *
from model import *

import numpy as np

if __name__ == '__main__':
  cd = CryptoCompareDataset('btc')
  cd.fetch()

  print(cd)
  # exit()
  # dl = DataloaderGenerator.gen(df, 1)
  # crypto_model = CryptoPredictorModel()
  # hp = HyperParams('ce', 'adam', 0.001, 3)
  # trained_model = Trainer.train(crypto_model, hp)
