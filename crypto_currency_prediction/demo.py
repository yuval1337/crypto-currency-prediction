import torch_loss
import torch_optim

from utils import *
from model_lib import *


def demo(from_file=False):
  gen = Generator(
      symbol='eth',
      to='usd',
      from_file=True
  )
  train_ds, valid_ds, test_ds = gen.split(train=70, valid=15, test=15)
  model = Model(gen.ds)
  if from_file:  # will load model weights from an existing .pth file
    model.load()
  hp = Hyperparams(
      loss_fn=torch_loss.mse_loss,
      epochs=100,
      batch_size=64,
      optimizer=torch_optim.Adam,
      learn_rate=0.01,
      # NOTE learning rate applies only to training and not to validation, which can result in inflated training loss
      reg_factor=0.0001,
      tolerance='high'
  )
  trainer = Trainer(model)
  trainer.run_train_session(train_ds, valid_ds, hp,
                            verbose=False, plot=False)
  trainer.run_test(test_ds)
  trainer.model.save()


if __name__ == '__main__':
  demo()
