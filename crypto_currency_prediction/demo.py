import torch

from utils import *
from model_lib import *


def demo():
  gen = Generator(
      symbol='btc',
      to='usd',
      from_file=True
  )
  train_set, test_set = gen.split(0.8)
  model = Model(gen.ds)
  hp = HyperParams(
      epochs=50,
      batch_size=8,
      optimizer=torch.optim.Adam,
      learn_rate=0.0001,
      reg_factor=0.01,
      stopper_tolerance='low'
  )
  trainer = Trainer(model)
  trainer.train(train_set,
                test_set,
                hp,
                verbose=True)


if __name__ == '__main__':
  demo()
