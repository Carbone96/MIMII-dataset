import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import layers

#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import learner_data_prep

class Autoencoder(Model):

  def __init__(self, InputSize : int):

    super(Autoencoder,self).__init__()
    
    self.encoder = tf.keras.Sequential([
      layers.Dense(64, activation="relu"),
      layers.Dense(64, activation="relu"),
      layers.Dense(8, activation="relu"),
    ])
    

    self.decoder = tf.keras.Sequential([
      layers.Dense(64, activation="relu"),
      layers.Dense(64, activation="relu"),
      layers.Dense(InputSize, activation="linear")])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

def foo(train_set,test_set):

  train_set, test_set, labels = learner_data_prep.foo(train_set, test_set)
  learnerModel = Autoencoder(InputSize= train_set.shape[1])
  learnerModel.compile(loss = "mse",
                        optimizer= 'adam')

  return train_set, test_set, labels, learnerModel


def main():
  

  import pandas as pd
  train_set = pd.DataFrame([[1,2,3], [1,2,3], [1,2,3], [1,2,3] , [1,2,3] , [1,2,3]])
  test_set = pd.DataFrame([ [1,2,3,0] , [1,2,3,0] ], columns=['1','2','3','labels' ])

  train_set, test_set, labels = learner_data_prep.foo(train_set, test_set)

  autoencoder =  Autoencoder(InputSize = train_set.shape[0]) 
  autoencoder.compile(loss = "mse",
                      optimizer = 'adam')

if __name__ == "__main__" : 
  main()
