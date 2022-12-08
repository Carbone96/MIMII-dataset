import pandas as pd
import numpy as np
import random

def HealthyData_separation(df_normal,df_abnormal):
    
    """ Separate the healthy dataset in two, such that for the TEST set there is a 50/50 proportion of Healthy & abnormal data """
    
    normal_data_length = df_normal.shape[0]  # get #raws
    abnormal_data_length = df_abnormal.shape[0] # get #raws
    train_data_selection =  np.array([1]*(normal_data_length - abnormal_data_length  ) + [0]*abnormal_data_length )
    train_data_selection = train_data_selection.astype(bool)
    random.shuffle(train_data_selection)
    
    train_set = df_normal[train_data_selection]
    healthy_test_set = df_normal[~train_data_selection]
    
    return train_set, healthy_test_set

def Testset_maker(healthy_test_set, df_abnormal):
    """ Create a randomized test set using the healthy test data and abnormal data """
    
    healthy_test_set = add_label_column(healthy_test_set, 'healthy')
    df_abnormal = add_label_column(df_abnormal, 'abnormal')
    
    Test_set_wLabels = pd.concat([healthy_test_set, df_abnormal], axis=0) #vertical concatenation of the two dataframes
    
    # Shuffle the test_set randomly
    Test_set_wLabels = Test_set_wLabels.sample(
            frac=1, # returns entire dataframe
            random_state=1, # makes the random reproducible
            ) #.reset_index()  # create news indexes
    
    # Get the 'labels' vector for the test_set
    test_labels = Test_set_wLabels['labels']
    test_labels = test_labels.astype(bool)
    
    # Retrieve the test_set datapoints
    Test_set_woLabels = Test_set_wLabels.loc[ : , Test_set_wLabels.columns != 'labels' ] # Fetch only the data (not the label!)
    
    return Test_set_woLabels, test_labels

def add_label_column(df, status : str):
    if status == 'healthy':
        ListValue = [0]
    elif status == 'abnormal':
        ListValue = [1]
        
    df.insert(loc = len(df.columns),
          column = 'labels',
          value = ListValue * df.shape[0])
    return df

def foo(DF_normal,DF_abnormal):
    
    train_set, healthy_test_set = HealthyData_separation(DF_normal,DF_abnormal) # create boolean vector for separation
    test_data, test_labels = Testset_maker(healthy_test_set, DF_abnormal)
    print('Train & test sets created !')

    return train_set, test_data, test_labels