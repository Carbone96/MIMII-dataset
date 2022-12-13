import pandas as pd
import tensorflow as tf
from tensorflow.keras import losses
import numpy as np
from abc import ABC, abstractmethod


class lossFunction(ABC):
    
    @abstractmethod
    def reconstructionEval(self,model,data):
        pass
    
class MSEloss(lossFunction):
    
    def reconstructionEval(self,model,data):
        data = tf.cast(data,float)
        reconstructions = model(data)
        return tf.keras.losses.mse(reconstructions,data)
    
class EvaluationMethod(ABC):
    
    @abstractmethod
    def AE_eval(autoencoder, test_set, eval_fun : lossFunction):
        return self.lossValues
   

class PSD_evaluation(EvaluationMethod):
    
    def AE_eval(self, autoencoder, test_set, eval_fun : lossFunction):
        
        return eval_fun.reconstructionEval(autoencoder,test_set)

class Spectro_evaluation(EvaluationMethod):
    
    def __init__(self,num_files, raws_per_file, columns_per_files):
        self.num_files = num_files
        self.raws_per_file = raws_per_file
        self.column_per_file = columns_per_files
        self.lossValues_list = []
        
    def AE_eval(self, autoencoder, test_set, eval_fun : lossFunction):
        """ Computes a lossValues for each file which undergo spectrogram processing, append and return the list for all files """
        for file_num in range(self.num_files):
            start_index, end_index = file_num*self.raws_per_file, (file_num+1)*self.raws_per_file
            test_set_slice = test_set.iloc[start_index:end_index]
            #lossValues = np.mean(eval_fun.reconstructionEval(autoencoder,test_set_slice))  
            lossValues = eval_fun.reconstructionEval(autoencoder,test_set_slice)

            self.lossValues_list.append(lossValues)
        return self.lossValues_list

def foo(method_name, autoencoder, test_set, dic_file : dict):
    """ Select the appropriate method for evaluating the dataframe depending on the method chosen """
    
    
    eval_fun = MSEloss()
 
    if method_name == 'psd':
        
        method_eval = PSD_evaluation()
        lossValues = method_eval.AE_eval(autoencoder, test_set, eval_fun)

        
    elif method_name =='spectro':
        
        num_files = dic_file['num_files']*2
        raws_per_file = dic_file['raws_per_file']
        columns_per_file = dic_file['columns_per_file']
        
        method_eval = Spectro_evaluation(num_files, raws_per_file, columns_per_file)
        lossValues = method_eval.AE_eval(autoencoder, test_set, eval_fun)
       
    elif method_name =='scalo':
        pass
    
    return lossValues

