from utils import *
from model_lib import *
import numpy as np


def demo():
  gen = Generator(
      symbol='btc',
      to='usd',
      from_file=True
  )
  train_set, test_set = gen.split(0.8)
  model_obj = Model(
      in_features=gen.ds.x_size,
      out_features=gen.ds.y_size,
      hidden_size=128,
      num_layers=2
  )
  hp_obj = HyperParams(
      epochs=50,
      batch_size=8,
      loss_func=torch.nn.functional.mse_loss,
      optimizer=torch.optim.Adam,
      lr=0.0001,
      reg_factor=0.01
  )
  trainer_obj = Trainer(model_obj, hp_obj, train_set)
  trainer_obj.train()

# def demo_graph():
#   gen = Generator('btc', 'usd', from_file=True)
#   plt = Plotter(gen)
#   plt.show_diff(pred)


if __name__ == '__main__':
  demo()
