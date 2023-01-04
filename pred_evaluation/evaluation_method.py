import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from dataclasses import dataclass
from abc import ABC, abstractmethod    

import pandas as pd
from typing import Union

def convert_to_numpy_array(data: any):
    if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
    if isinstance(data, tf.Tensor):
            data = data.numpy()
    return data

class EvaluationMethodClass(ABC):

    @abstractmethod
    def fit(self, training_data):
        """ Calibrate the parameters for the evaluation based on the training data"""
        pass

    @abstractmethod
    def transform(self, testing_data):
        """ Transform the data based on the calibrated parameters """
        pass

    @abstractmethod
    def compute_anomaly_score(self, testing_data):
        """ Perform the evaluation on the testing data"""
        pass


@dataclass
class mahalanobisClass(EvaluationMethodClass):

    meanVal = None
    inv_cov = None

    def fit(self,training_data):
        """ Fit the training error"""
        training_data = convert_to_numpy_array(training_data)
        self.meanVal = np.mean(training_data, axis=0)
        self.inv_cov = np.linalg.inv(np.cov(training_data.T))
        #self.cov = np.cov(training_data.T)

    def transform(self,test_data):
        """ Transform the test error """
        test_data = convert_to_numpy_array(test_data)
        data_mu = test_data - self.meanVal
        left = np.dot(data_mu , self.inv_cov)
        distance = np.dot(left, data_mu.T)
        return distance.diagonal()

    def compute_anomaly_score(self, test_data):
        """ Return an anomaly score which is the Root Mean Square Value of the different mahalanobis distance of the features"""
        test_data = convert_to_numpy_array(test_data)

        return np.sum(np.square(self.transform(self,test_data)))
        #return np.sqrt(np.mean(np.square(self.transform(self,test_data))))


@dataclass
class SSEClass(EvaluationMethodClass):

    """ Sum of Squared errors"""

    def fit(self,training_data):
        pass
    def transform(self, test_data):
        return test_data

    def compute_anomaly_score(self,test_data):
        """ Return an anomaly score which is the Root Mean Square Value of the features"""
        test_data = convert_to_numpy_array(test_data)
        return np.sum(np.square(self.transform(self,test_data)))

@dataclass
class rmseClass(EvaluationMethodClass):

    def fit(self,training_data):
        pass
    def transform(self, test_data):
        return test_data

    def compute_anomaly_score(self,test_data):
        """ Return an anomaly score which is the Root Mean Square Value of the features"""
        test_data = convert_to_numpy_array(test_data)
        return np.sqrt(np.mean(np.square(self.transform(self,test_data))))


def factory_evaluation_method(method_selected : str, training_data : Union[pd.DataFrame, tf.Tensor]):

    """ Initialize the evaluation method. It can be used to transform afterwards. """

    dict_method_evaluation = {'rmse' : rmseClass,
       'mahalanobis' : mahalanobisClass,
       'sse' : SSEClass}

    method_selected = dict_method_evaluation[method_selected]
    return method_selected



def main():    
    # Generate sample data
    data = np.random.rand(100,2)
    true = np.random.rand(100,2)

    learner = Mahalanobis()

    # Fit the model with the data

    learner.fit(data)

    # Transform the data
   
    print(learner.transform(data, true))
    print(len(data))

if __name__ == "__main__":
    main()
