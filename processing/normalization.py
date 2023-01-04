import numpy as np
import pandas as pd
from typing import Union

def get_global_min_max(df_list : list[pd.DataFrame]) -> tuple[float, float]:
    """
    Get the global minimum and maximum values of the dataframes in the given list.

    Parameters:
    df_list (List[pd.DataFrame]): A list of dataframes.

    Returns:
    Tuple[float, float]: A tuple containing the global minimum and maximum values.
    """
    df = pd.concat(df_list)
    global_min = df.min().min()
    global_max = df.max().max()
    return global_min, global_max

def min_max_normalization(df1 : pd.DataFrame, df2 : pd.DataFrame, global_min : float, global_max : float, factor : float = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform Min-Max normalization on the two dataframes.

    Parameters:
    df1 (pd.DataFrame): First dataframe.
    df2 (pd.DataFrame): Second dataframe.
    global_min (float): Global minimum value.
    global_max (float): Global maximum value.
    factor (float, optional): Normalization factor. Default is 1.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame, float, float]: A tuple containing the normalized dataframes, and the global minimum and maximum values.
    """
    if global_min == global_max:
        raise ValueError("Global minimum and maximum values are equal.")
    df1_normalized = df1.apply(lambda x : (x - global_min) / (global_max - global_min) * factor)
    df2_normalized = df2.apply(lambda x : (x - global_min) / (global_max - global_min) * factor)

    return df1_normalized, df2_normalized

def normalize(df_normal: pd.DataFrame, df_abnormal: pd.DataFrame, hyper_param: dict):

    global_min, global_max = get_global_min_max([df_normal,df_abnormal])

    df_normal_normalized, df_abnormal_normalized= min_max_normalization(df_normal, df_abnormal,global_min, global_max)

    hyper_param.update(({"original_min": global_min, "original_max": global_max}))

    return df_normal_normalized,df_abnormal_normalized, hyper_param


def main():

    # list of lists
    list_test_1 = np.array([[1,12,4,],[1,12,4],[1,12,4]])
    list_test_2 = np.array([[[1,12],[4,64],[74,654]],[[1,12],[4,64],[74,654]]])

    hyper_param_1= {'raws_per_file' : 1,
                    'method_name' : 'psd'}
    hyper_param_2 = {'raws_per_file' : 2,
                    'method_name' : 'spectro'}



    df_normal, df_abnormal, hyper_param = normalize_data(list_test_1, list_test_1, hyper_param_1)


    print(df_abnormal)

if __name__ == "__main__":
    main()