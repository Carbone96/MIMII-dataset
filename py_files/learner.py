import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import layers

import mahlanobis_dist
import learner_data_prep

class Autoencoder(Model):

  def __init__(self, InputSize : int, latent_dim : int):

    super(Autoencoder,self).__init__()

    self.InputSize = InputSize
    self.latent_dim = latent_dim

    self.encoder = tf.keras.Sequential([
      layers.Dense(64, activation="relu"),
      layers.Dense(64, activation="relu"),
      layers.Dense(self.latent_dim, activation="relu"),
    ])
    

    self.decoder = tf.keras.Sequential([
      layers.Dense(64, activation="relu"),
      layers.Dense(64, activation="relu"),
      layers.Dense(self.InputSize, activation="linear")])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


def createLearner(train_set,test_set,hyper_param, EPOCHS = 50, BATCH  = 512, latent_dim = 8, loss_fun = "mse"):

  if hyper_param['learner'] == 'autoencoder':
    train_set, test_set,labels = learner_data_prep.foo(train_set, test_set)
    learnerModel = Autoencoder(InputSize= train_set.shape[1], latent_dim = latent_dim)
    if loss_fun == "mse":
      learnerModel.compile(loss = "mse",
                            optimizer= 'adam')               
    if loss_fun == "mahalanobis":
      learnerModel.compile(loss = mahlanobis_dist.fit,
                            optimizer= 'adam')
    print(f'Learner will fit the data now !')
    learnerModel.fit(train_set, train_set, 
                                  epochs=EPOCHS, 
                                  batch_size=BATCH,
                                  validation_data=(test_set,test_set),
                                  validation_split = 0.1,
                                  verbose = 0,
                                  shuffle=False # you don't want to shuffle here, so that you keep track of the labels !
                                 )
    print(f'Learner has trained. Ready to predict, boss !')
  return learnerModel




def main():
  
  import pandas as pd
  train_set = pd.DataFrame([[1,2,3], [1,2,3], [1,2,3], [1,2,3] , [1,2,3] , [1,2,3]])
  test_set = pd.DataFrame([ [1,2,3,0] , [1,2,3,0] ], columns=['1','2','3','labels' ])

  train_set, test_set, labels = learner_data_prep.foo(train_set, test_set)

  autoencoder =  Autoencoder(InputSize = train_set.shape[0], latent_dim = 5) 
  autoencoder.compile(loss = "mse",
                      optimizer = 'adam')

if __name__ == "__main__" : 
  main()
