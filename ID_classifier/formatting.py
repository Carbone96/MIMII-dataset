import numpy as np
import pandas as pd

def reshape(df : pd.DataFrame, file_count : int, raws_per_file: int):
    final_np_df = df.to_numpy().reshape(file_count, raws_per_file, df.shape[1], 1)
    return final_np_df

def CNNformat(train_set : pd.DataFrame, test_set : pd.DataFrame, raws_per_file : int):
    # Convert that segment into a numpy array
    file_count_normal = int(train_set.shape[0]/raws_per_file)
    file_count_abnormal = int(test_set.shape[0]/raws_per_file)
    return reshape(train_set,file_count_normal,raws_per_file), reshape(test_set,file_count_abnormal,raws_per_file)



def test_reshape():
    df = pd.DataFrame({'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                       'B': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]})
    file_count = 2
    raws_per_file = 5

    result = reshape(df, file_count, raws_per_file)
    expected_shape = (file_count, raws_per_file, df.shape[1], 1)
    assert result.shape == expected_shape, f'Expected shape {expected_shape}, but got {result.shape}'

    expected_first_part = np.array([[[1], [10]],
                                     [[2], [9]],
                                     [[3], [8]],
                                     [[4], [7]],
                                     [[5], [6]]])
    assert np.allclose(result[0], expected_first_part), f'Expected {expected_first_part}, but got {result[0]}'

    expected_second_part = np.array([[[6], [5]],
                                      [[7], [4]],
                                      [[8], [3]],
                                      [[9], [2]],
                                      [[10], [1]]])
    assert np.allclose(result[1], expected_second_part), f'Expected {expected_second_part}, but got {result[1]}'


def main():
    test_reshape()

if __name__ == "__main__":
    main()