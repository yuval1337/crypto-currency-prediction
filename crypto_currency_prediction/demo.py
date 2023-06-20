from utils import *
from model_lib import *
from glob import glob

if __name__ == '__main__':
  dsg_obj = DatasetGenerator(
      symbol='btc',
      to='usd',
      from_file=True
  )

  train_set, test_set = dsg_obj.split(0.8)

  model_obj = Model(
      input_size=dsg_obj.ds.get_x.shape[2],
      hidden_size=128,
      num_layers=2
  )

  hp = HyperParams(
      epochs=30,
      batch_size=64,
      loss_func=torch.nn.functional.mse_loss,
      optimizer=torch.optim.Adam,
      lr=0.0001,
      reg_factor=0.001
  )

  model_obj_trained = train(model_obj, hp, train_set)
  test(model_obj_trained, hp, test_set)
