from utils import *
from model_lib import *


if __name__ == '__main__':
  dsg_obj = DatasetGenerator()
  dsg_obj.fetch(symbol='btc')  # or dsg_obj.read('{symbol?}_{timestamp?}.csv')

  train_set, test_set = dsg_obj.split(0.8)

  model_obj = Model(
      input_size=dsg_obj.ds.get_X.shape[2],
      hidden_size=64,
      num_layers=3
  )

  hp = HyperParams(
      epochs=100,
      batch_size=64,
      loss='mse',
      optimizer='adam',
      lr=0.0001
  )

  model_obj_trained = train(model_obj, hp, train_set)
