import tensorflow as tf
import numpy as np
from learner import Autoencoder
from mahlanobis_dist import Mahalanobis
import pandas as pd
from typing import Optional, Callable


def get_reconstructed_labels(hyper_param: dict):
    """
    Generate and return a list of labels with the same number of 0s and 1s as the number of abnormal and normal files.
    """
    return [0] * hyper_param['file_count_abnormal'] + [1] * hyper_param['file_count_abnormal']

def split_test_and_labels(df: pd.DataFrame):
    """
    Split the given DataFrame into data and labels.
    """
    # Get the 'labels' vector for the test_set
    labels = df['labels']
    labels = df.astype(bool)
    # Retrieve the test_set datapoints
    data = df.loc[:, df.columns != 'labels']  # Fetch only the data (not the label!)
    return data, labels

def mahala_loss(train_data : pd.DataFrame, test_data : pd.DataFrame, hyper_param : dict, model : Optional[any] = None ):
    """
    Calculate and return the mahalanobis loss for the given data and model, using the given Mahalanobis instance.
    """
    loss_fun = Mahalanobis()
    # With the model : use the reconstruction erros to build the normal distribution
    if model is not None:
        # Fit train errors
        train_data = tf.cast(train_data,float)
        reconstructions = model(train_data)
        train_errors = train_data - reconstructions
        loss_fun.fit(train_errors)

        # Compute anomaly score of test errors
        test_data = tf.cast(test_data, float)
        reconstructions = model(test_data)
        test_error = test_data - reconstructions
        return loss_fun.np_transform(test_error)

    # Without a model : use the data as-is
    loss_fun.np_fit(train_data)  # Fit train data
    return loss_fun.np_transform(test_data) # Compute anomaly score of test data


def mse_loss(model, data):
    """
    Calculate and return the mean squared error loss for the given model and data.
    """
    data = tf.cast(data, float)
    errors = data - model(data)
    return np.sum(np.square(errors.numpy()))


def spectro_evaluation(autoencoder: Autoencoder, loss_func: Optional[Callable], test_set: pd.DataFrame, hyper_param: dict):
    """
    Evaluate the error corresponding to one file in the spectro method.
    """
    test_set, labels = split_test_and_labels(test_set)
    
    raws_per_file = hyper_param['raws_per_file']
    file_count_test_set = hyper_param['file_count_abnormal'] * 2  #  50/50 normal & abnormal data

    anomaly_score_list = []

    for file_num in range(file_count_test_set):
        test_set_slice = test_set.iloc[file_num * raws_per_file: (file_num + 1) * raws_per_file]
        test_set_slice = tf.cast(test_set_slice, float)
        reconstructions = autoencoder(test_set_slice)
        errors = test_set_slice - reconstructions
        # Use the function passed as an argument to compute the anomaly score based on errors
        anomaly_score = loss_func(errors)
        anomaly_score_list.append(anomaly_score)
    return anomaly_score_list

    



def psd_evaluation(autoencoder: Autoencoder, loss_fun: Mahalanobis, test_set: pd.DataFrame, train_set: pd.DataFrame):
    """
    Evaluate the error corresponding to the psd method.
    """
    test_set, labels = split_test_and_labels(test_set)
    if loss_fun is None:
        return mse_loss(test_set, autoencoder)
    return loss_fun(test_set, train_set, autoencoder)


def evaluate(autoencoder: Autoencoder, train_set: pd.DataFrame, test_set: pd.DataFrame, hyper_param: dict):
    """
    Select the appropriate method for evaluating the dataframe depending on the method chosen.
    """
    method_name = hyper_param['method_name']

    methods = {
        'psd': psd_evaluation,
        'spectro': spectro_evaluation
    }

    #look for the selected method
    evaluation_func = methods[method_name]
    
    return evaluation_func(autoencoder, train_set, test_set, hyper_param), get_reconstructed_labels(hyper_param)