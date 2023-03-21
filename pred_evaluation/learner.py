import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers

import pandas as pd

class Autoencoder(Model):

  def __init__(self, InputSize : int, latent_dim : int, intermediate_layer : int):

    super(Autoencoder,self).__init__()

    self.InputSize = InputSize
    self.latent_dim = latent_dim
    self.intermediate_layer = intermediate_layer

    self.encoder = tf.keras.Sequential([
      layers.Dense(self.intermediate_layer, activation="relu"),
      layers.Dense(self.intermediate_layer, activation="relu"),
      layers.Dense(self.latent_dim, activation="relu"),
    ])
    
    self.decoder = tf.keras.Sequential([
      layers.Dense(self.intermediate_layer, activation="relu"),
      layers.Dense(self.intermediate_layer, activation="relu"),
      layers.Dense(self.InputSize, activation="linear")])

  def call(self, input_data):
    """ Returns the reconstructed data """
    encoded = self.encoder(input_data)
    decoded = self.decoder(encoded)
    return decoded

  def reconstructions_error(self, x):
    """ Returns the reconstruction error """
    x = tf.cast(x, float)
    reconstructions = self.call(x)
    return reconstructions - x

def createLearner(train_set: pd.DataFrame,test_set: pd.DataFrame, labels : pd.DataFrame, learner_dict : dict):

  loss_fun = learner_dict['loss_fun']
  latent_dim = learner_dict['latent_dim']
  learner_name = learner_dict['name']
  epochs = learner_dict['epochs']
  batch_size = learner_dict['batch_size']

  if learner_name == 'autoencoder':
    autoencoder = Autoencoder(InputSize= train_set.shape[1], latent_dim = latent_dim, intermediate_layer = 64)
    autoencoder.compile(loss = loss_fun,optimizer= 'adam')
  if learner_name == 'autoencoder_sparse':
    autoencoder = Autoencoder(InputSize= train_set.shape[1], latent_dim = latent_dim, intermediate_layer = train_set.shape[1])
    autoencoder.compile(loss = loss_fun,optimizer= 'adam')


    print(f'Autoencoder will fit the data now !')
    autoencoder.fit(train_set, train_set, 
                                  epochs=epochs, 
                                  batch_size=batch_size,
                                  validation_data=(test_set,test_set),
                                  verbose = 0,
                                  shuffle=True
                                 )
    print(f'Autoencoder has trained. Ready to predict, boss !')
    return autoencoder




def main():
  
  import pandas as pd
  train_set = pd.DataFrame([[1,2,3], [1,2,3], [1,2,3], [1,2,3] , [1,2,3] , [1,2,3]])
  test_set = pd.DataFrame([ [1,2,3,0] , [1,2,3,0] ], columns=['1','2','3','labels' ])

  #train_set, test_set, labels = learner_data_prep.foo(train_set, test_set)

  autoencoder =  Autoencoder(InputSize = train_set.shape[0], latent_dim = 5) 
  autoencoder.compile(loss = "mse",
                      optimizer = 'adam')

if __name__ == "__main__" : 
  main()
