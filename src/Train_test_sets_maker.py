import pandas as pd
import numpy as np
import random

def NormalData_separation(NORMAL_data_length,ABNORMAL_data_length):
    
    # Create a randomized boolean vector to separe the normal DF in two datasets.
    # INPUT : 2 df + # of rows
    # OUTPUT : boolean vector
    train_data_selection =  np.array([1]* (NORMAL_data_length - ABNORMAL_data_length  ) + [0] * ABNORMAL_data_length )
    random.shuffle(train_data_selection)
    return train_data_selection.astype(bool)


def fun(DF_normal,DF_abnormal):
    
    # Requires : Spectro_DF_maker.py
    
    # Create the train set (normal data) & test set (normal data + abnormal data)
    # INPUT : normal df + abnormal df
    # OUTPUT : train set + test set + test labels (to evaluate predictions)
    # ----------------------------------------------------------------------------
    
    normal_data_length = DF_normal.shape[0]  # get #raws
    abnormal_data_length = DF_abnormal.shape[0] # get #raws
    
    train_data_selection = NormalData_separation(normal_data_length,abnormal_data_length) # create boolean vector for separation
    
    train_set = DF_normal[train_data_selection]
    normal_test_data = DF_normal[~train_data_selection] # healthy data for the test_set
    
    
    ## Adds 'labels' column in Pythonic manner
    DF_abnormal.insert(loc = len(DF_abnormal.columns),
          column = 'labels',
          value = [0] * abnormal_data_length)
    normal_test_data.insert(loc = len(normal_test_data.columns),
          column = 'labels',
          value = [1] * abnormal_data_length)
    
    # Create the test_set
    test_set = pd.concat([DF_abnormal, normal_test_data], axis=0) #vertical concatenation of the two dataframes
    # Shuffle the test_set randomly
    test_set = test_set.sample(
        frac=1, # returns entire dataframe
        random_state=1, # makes the random reproducible
    ) #.reset_index()  # create news indexes
    
    # Get the 'labels' vector for the test_set
    test_labels = test_set['labels']
    test_labels = test_labels.astype(bool)
    
    # Retrieve the test_set datapoints
    test_data = test_set.loc[ : , test_set.columns != 'labels' ] # Fetch only the data (not the label!)
    print('Train & test sets created !')
    
    return train_set, test_data, test_labels