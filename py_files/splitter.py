import pandas as pd
import numpy as np
import random

""" 

    UPDATE TO DO: 
    The random part is uncontrolled
    ---> it may be interresting to include a seed such that experiment are repeatable !

"""

def NormalData_separation(df_normal : pd.DataFrame , hyper_param :dict):
    """ Select randomly which normal experiments will be part of the train set"""

    normal_data_count = hyper_param['file_count_normal'] * hyper_param['raws_per_file'] 
    abnormal_data_count = hyper_param['file_count_abnormal'] * hyper_param['raws_per_file']

    train_data_selection =  np.array([1]*(normal_data_count - abnormal_data_count) + [0]*abnormal_data_count)
    train_data_selection = train_data_selection.astype(bool)
    random.shuffle(train_data_selection)
        
    train_set = df_normal[train_data_selection]
    normal_part_test_set = df_normal[~train_data_selection]
        
    return train_set, normal_part_test_set

def add_label_column(df : pd.DataFrame, status : str):
    """ Adds a label column to a pd.DataFrame """
    if status == 'normal':
        df.insert(loc = len(df.columns),column = 'labels', value = [1]*df.shape[0])
        return  df

    df.insert(loc = len(df.columns),column = 'labels', value = [0]*df.shape[0])
    return df


def Testset_maker(normal_part_test_set : pd.DataFrame, df_abnormal: pd.DataFrame):
    """ Create a randomized test set using the healthy test data and abnormal data """
    
    normal_part_test_set = add_label_column(normal_part_test_set, 'normal')
    df_abnormal = add_label_column(df_abnormal, 'abnormal')

    Test_set_wLabels = pd.concat([normal_part_test_set, df_abnormal], axis=0) #vertical concatenation of the two dataframes
    

    return Test_set_wLabels

def foo(df_normal : pd.DataFrame, df_abnormal : pd.DataFrame, hyper_param : dict):
    """ Compute the train & test sets + test labels"""
    train_set , normal_part_test_set = NormalData_separation(df_normal, hyper_param)
    test_set = Testset_maker(normal_part_test_set, df_abnormal)
    return train_set, test_set




def Test_df_maker():
    """ function that tests the implementation """

    listTest_normal = [[1,2,3],[1,2,3],[1,2,3],[1,2,3]]
    listTest_abnormal = [[4,5,6]]

    hyper_param_test = {'file_count_normal' : 4,
                        'file_count_abnormal' : 1,
                        'raws_per_file': 1}

    df_test_normal = pd.DataFrame(listTest_normal)
    df_test_abnormal = pd.DataFrame(listTest_abnormal)

    train_set, normal_part_test_set = NormalData_separation(df_test_normal,hyper_param_test)

    print(f'The train set is {train_set}')
    print(f'The normal part set is {normal_part_test_set}')

    test_set= Testset_maker(normal_part_test_set, df_test_abnormal)

    print(f'The train set is {test_set}')


    print('Train & test sets created !')


def main():
    Test_df_maker()

if __name__ == "__main__":
    main()