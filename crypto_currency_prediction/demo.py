import torch

from utils import *
from model_lib import *


def demo():
  gen = Generator(
      symbol='btc',
      to='usd',
      from_file=True
  )
  train_set, valid_set = gen.split(0.7)
  model = Model(gen.ds)
  hp = HyperParams(
      epochs=50,
      batch_size=16,
      optimizer=torch.optim.RMSprop,
      learn_rate=0.0001,
      reg_factor=0.1,
      stopper_tolerance='high'
  )
  trainer = Trainer(model)
  trainer.run_train_session(train_set,
                            valid_set,
                            hp,
                            verbose=True)


if __name__ == '__main__':
  demo()
