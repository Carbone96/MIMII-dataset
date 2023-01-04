import pandas as pd
import numpy as np
import random



def create_train_and_normal_test_sets(df_normal : pd.DataFrame , hyper_param :dict) -> tuple([pd.DataFrame, pd.DataFrame]):
    """
    Select a random subset of the normal experiments to be included in the train set.

    Parameters:
    - df_normal (pd.DataFrame): Dataframe containing normal experiments.
    - hyper_param (dict): Dictionary of hyperparameters.

    Returns:
    - tuple: Tuple containing two dataframes: the train set and the normal part of the test set.
    """

    total_file_numbers = hyper_param['file_count_normal']
    file_count_train_set = hyper_param['file_count_normal'] - hyper_param['file_count_abnormal']
    random_indices_number_selection = random.sample(range(total_file_numbers), file_count_train_set)
    random_indices_number_selection = [val*hyper_param['raws_per_file'] for val in random_indices_number_selection]
    random_indices_number_selection = [val + i for val in random_indices_number_selection for i in range(hyper_param['raws_per_file'])]

    train_set = df_normal.iloc[random_indices_number_selection]
    normal_part_test_set = df_normal.drop(random_indices_number_selection)



    return train_set, normal_part_test_set
    


def create_test_set_with_labels(normal_part_test_set : pd.DataFrame, df_abnormal: pd.DataFrame) -> pd.DataFrame:
    """
    Create a test set with labels by sampling from the normal part of the test set and the abnormal dataframe.

    Parameters:
    - normal_part_test_set (pd.DataFrame): Normal part of the test set.
    - df_abnormal (pd.DataFrame): Dataframe containing abnormal experiments.

    Returns:
    - pd.DataFrame: Test set with labels.
    """

    normal_part_test_set = normal_part_test_set.assign(labels=1)
    df_abnormal = df_abnormal.assign(labels=0)

    test_set = pd.concat([normal_part_test_set, df_abnormal], ignore_index=True)
    
    return test_set


def split(df_normal : pd.DataFrame, df_abnormal : pd.DataFrame, hyper_param : dict) -> tuple([pd.DataFrame, pd.DataFrame]):
    """
    Split the data into train and test sets.
    """

    random.seed(42)

    train_set, normal_part_test_set = create_train_and_normal_test_sets(df_normal, hyper_param)
    test_set = create_test_set_with_labels(normal_part_test_set, df_abnormal)

    return train_set, test_set


### Testing part 

def main():
    """
    Test the implementation.
    """
    list_test_normal = [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]
    list_test_abnormal = [[4, 5, 6]]

    hyper_param_test = {'file_count_normal': 4,
                        'file_count_abnormal': 1,
                        'raws_per_file': 1}

    df_test_normal = pd.DataFrame(list_test_normal)
    df_test_abnormal = pd.DataFrame(list_test_abnormal)

    train_set, normal_part_test_set = create_train_and_normal_test_sets(df_test_normal, hyper_param_test)

    print(f'The train set is {train_set}')
    print(f'The normal part set is {normal_part_test_set}')

    test_set = create_test_set_with_labels(normal_part_test_set, df_test_abnormal)

    print(f'The test set is {test_set}')
    print('Train & test sets created!')

if __name__ == "__main__":
    main()