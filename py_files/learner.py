import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend as K

from tensorflow.keras.models import Model
from tensorflow.keras import layers

#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from py_files import learner_data_prep

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

def foo(train_set,test_set,hyper_param, EPOCHS = 50, BATCH  = 512):

  if hyper_param['learner'] == 'autoencoder':
    train_set, test_set, labels = learner_data_prep.foo(train_set, test_set)
    learnerModel = Autoencoder(InputSize= train_set.shape[1])
    learnerModel.compile(loss = "mse",
                          optimizer= 'adam')
    learnerModel.fit(train_set, train_set, 
                                  epochs=EPOCHS, 
                                  batch_size=BATCH,
                                  validation_data=(test_set,test_set),
                                  validation_split = 0.1,
                                  verbose = 0,
                                  shuffle=False # you don't want to shuffle here, so that you keep track of the labels !
                                 )

  return learnerModel


def mahala_dist(y_pred, y_true):
    a = y_pred - y_true # 1xInputSize
    cov = tfp.stats.covariance(tf.transpose(a)) # Inputsize x Inputsize
    mull = K.dot(tf.linalg.inv(cov), a)
    mull2 = K.dot(mull,tf.transpose(a))
    dist = tf.sqrt(tf.math.abs(mull2))
    return dist


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
