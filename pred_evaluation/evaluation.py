import tensorflow as tf
import numpy as np
import pandas as pd
from typing import Optional

from learner import Autoencoder
import evaluation_method
from evaluation_method import EvaluationMethodClass



def AE_evaluation(autoencoder: Autoencoder, eval_method: EvaluationMethodClass, test_set: pd.DataFrame, file_count : int,raws_per_file :int):
    """
    Evaluate the error corresponding to one file (that might be spread on multiple raws in the test set).
    """
    anomaly_score_list = []
    for file_num in range(file_count):
        """
        WARNING : make sure the line below does not give index error (out of bounds)
        """
        test_set_slice = test_set.iloc[file_num * raws_per_file: (file_num + 1) * raws_per_file]
        test_set_slice = tf.cast(test_set_slice, float)
        # Use the function passed as an argument to compute the anomaly score based on errors
        anomaly_score = eval_method.compute_anomaly_score(eval_method,autoencoder.reconstructions_error(test_set_slice))
        anomaly_score_list.append(anomaly_score)
    return anomaly_score_list

    

def evaluate(autoencoder: Autoencoder, train_set: pd.DataFrame, test_set: pd.DataFrame,labels : pd.DataFrame , settings : dict):
    """
    Select the appropriate method for evaluating the dataframe depending on the method chosen.
    """

    raws_per_file = settings['raws_per_file']
    method_selected = settings['evaluation_method']
    
    
    file_count = len(labels)

    # Prepare evaluation method:
    eval_method = evaluation_method.factory_evaluation_method(method_selected= method_selected, training_data =train_set)
    errors = autoencoder.reconstructions_error(train_set)
    eval_method.fit(eval_method,errors) # calibrate the evaluation method

    return AE_evaluation(autoencoder, eval_method, test_set, file_count, raws_per_file)