import pandas as pd
import tensorflow as tf
from learner import Autoencoder
import numpy as np

from mahlanobis_dist import Mahalanobis

"""
    UPDATE TO DO :

    Rework the implementation such that test_set that need no reshape and those who do gets call in a more generic way
    ---> Makes it easier to see what is the data transformation


"""



def ReconstructLabels(hyper_param:dict):
    return [0]*hyper_param['file_count_abnormal'] + [1]*hyper_param['file_count_abnormal']

def SplitTestAndLabels(df: pd.DataFrame):
    # Get the 'labels' vector for the test_set
    labels = df['labels']
    labels = df.astype(bool)
    # Retrieve the test_set datapoints
    data = df.loc[ : , df.columns != 'labels'] # Fetch only the data (not the label!)
    return data, labels

def MSEloss(model, data):
    data = tf.cast(data,float)
    reconstructions = model(data)
    return tf.keras.losses.mse(reconstructions,data)

def MahalaLoss(data : pd.DataFrame,model, lossFun:Mahalanobis):
    """ Apply the mahalanobis distance without reconstructing the data """
    
    
    data = tf.cast(data,float)
    reconstructions = model(data)
    error = data - reconstructions
    
    return lossFun.np_transform(error)

def MahalaDirect(data, lossFun : Mahalanobis):
    return lossFun.transform(data)

def SpectroEvaluation(autoencoder : Autoencoder , lossFun: Mahalanobis, test_set : pd.DataFrame,  hyper_param : dict):
    """
    Evaluate the error corresponding to one file
    """
    test_set, labels = SplitTestAndLabels(test_set)
    
    raws_per_file = hyper_param['raws_per_file']
    file_count_test_set = hyper_param['file_count_abnormal']*2  #  50/50 normal & abnormal data

    lostValues_list = []
    for file_num in range(file_count_test_set):
        test_set_slice = test_set.iloc[file_num*raws_per_file : (file_num+1)*raws_per_file]
        test_set_slice = tf.cast(test_set_slice,float)
        reconstructions = autoencoder(test_set_slice)
        errors = test_set_slice - reconstructions
        lossValues = np.sum(lossFun.np_transform(errors))
        lostValues_list.append(lossValues)
    return lostValues_list, ReconstructLabels(hyper_param)

def PSDEvaluation(autoencoder : Autoencoder , lossFun : Mahalanobis, test_set : pd.DataFrame, hyper_param : dict):

    test_set, labels = SplitTestAndLabels(test_set)

    return MahalaLoss(test_set,autoencoder,lossFun), ReconstructLabels(hyper_param)


def foo(autoencoder : Autoencoder,  train_set : pd.DataFrame, test_set : pd.DataFrame, hyper_param : dict):
    """ Select the appropriate method for evaluating the dataframe depending on the method chosen """
    if hyper_param['method_name'] == 'psd':
        lossFun = Mahalanobis()
        tf_train_set = tf.cast(train_set,float)
        error = tf_train_set - autoencoder(tf_train_set)
        lossFun.np_fit(error)
        return PSDEvaluation(autoencoder,lossFun,test_set,hyper_param)
    if hyper_param['method_name'] =='spectro':
        lossFun = Mahalanobis()
        tf_train_set = tf.cast(train_set,float)
        error = tf_train_set - autoencoder(tf_train_set)
        lossFun.np_fit(error)
        return SpectroEvaluation(autoencoder,lossFun,test_set,hyper_param)
    if hyper_param['method_name'] == 'scalo':
        return PSDEvaluation(autoencoder,test_set,hyper_param)

def main():
    
    df_test = pd.DataFrame([[1,2], [3,4]], columns = ['1' , 'labels'])
    print(df_test)
    print(df_test['labels'])

    data = df_test.loc[ : , df_test.columns != 'labels']
    print(data)

if __name__ == '__main__':
    main()