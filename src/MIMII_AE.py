import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses

class AnomalyDetector(Model):
    
  def __init__(self,InputSize):
    super(AnomalyDetector, self).__init__()
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

def fun(InputSize, loss_fun = "mse"):
    
    AE =  AnomalyDetector(InputSize)
    AE.compile(loss=loss_fun,
               optimizer='adam'
              )
    return AE
