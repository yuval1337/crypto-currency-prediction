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
      input_size=gen.ds.get_x.shape[2],
      hidden_size=128,
      num_layers=2
  )

  hp = HyperParams(
      epochs=100,
      batch_size=64,
      loss_func=torch.nn.functional.mse_loss,
      optimizer=torch.optim.Adam,
      lr=0.00001,
      reg_factor=0.01
  )

  model_obj_trained = train(model_obj, hp, train_set)

  close_pred = test(model_obj_trained, hp, test_set)


# def demo_graph():
#   gen = Generator('btc', 'usd', from_file=True)
#   plt = Plotter(gen)
#   plt.show_diff(pred)


if __name__ == '__main__':
  demo()
