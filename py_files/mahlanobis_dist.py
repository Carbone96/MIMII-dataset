import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def fit(y_true, y_pred):
    """Fit the Mahalanobis distance model
    Parameters
        ----------
    data : np.ndarray or pd.DataFrame
    Data to fit the model
        """ 
    data = y_pred - y_true
    data_tf = tf.convert_to_tensor(data, dtype=tf.float32)
    meanVal = tf.reduce_mean(data_tf, axis=0)
    inv_cov = tf.linalg.inv(tfp.stats.covariance(data_tf))
    data_mu = data_tf - meanVal

    left = tf.matmul(data_mu, inv_cov)
    distance = tf.matmul(left, tf.transpose(data_mu))
    distance_diag = tf.linalg.diag_part(distance)
    return distance_diag

def LossCalc(y_true, y_pred):
    data = y_pred - y_true
    data_tf = tf.convert_to_tensor(data, dtype=tf.float32)
    meanVal = tf.reduce_mean(data_tf, axis=0)
    inv_cov = tf.linalg.inv(tfp.stats.covariance(data_tf))
    data_mu = data_tf - meanVal

    left = tf.matmul(data_mu, inv_cov)
    distance = tf.matmul(left, tf.transpose(data_mu))
    distance_diag = tf.linalg.diag_part(distance)
    return distance_diag

def main():    
    # Generate sample data
    data = np.random.rand(100,2)
    true = np.random.rand(100,2)

    
    # Fit the model with the data
    dist=fit(data,true)

    # Transform the data
   
    print(dist)
    print(len(dist))

if __name__ == "__main__":
    main()
