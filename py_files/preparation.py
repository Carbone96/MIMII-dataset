import numpy as np
import pandas as pd

from abc import ABC,abstractclassmethod


""""
REFACTORING : 

    TOO MUCH DATAFRAME TO HANDLE
    ---> delete the flat one !
It is much better to create only the original dataframe as intended by the method. 
The number of raws per experiment can be used to reconstruct experiments later on !


"""



class MinMax():

    def __init__(self,factor =1):
        self.factor = factor
        self.minVal = None
        self.maxVal = None

    def compute(self, df1, df2):
        """ Minmax normalisation comparing 2 dataframes : data values € [0,1] * factor """
        self.MinMax_selector(df1,df2)
        df1 = ((df1 - self.minVal) / (self.maxVal - self.minVal)) * self.factor
        df2 = ((df2 - self.minVal) / (self.maxVal - self.minVal)) * self.factor
        return df1,df2

    def getMin_fromDF(self,df):
        """ Return the minimum of a df as a float"""    
        raws_min = df.min(axis=0)
        global_min = raws_min.min()
        return float(global_min)

    def getMax_fromDF(self,df):
        """ Return the minimum of a df as a float"""    
        raws_min = df.max(axis=0)
        global_min = raws_min.max()
        return float(global_min)

    def MinMax_selector(self,df1,df2):
        """ Find the minimum and maximum values between 2 dataframes"""
        min_normal,max_normal = self.getMin_fromDF(df1), self.getMax_fromDF(df1)
        min_abnormal,max_abnormal = self.getMin_fromDF(df1), self.getMax_fromDF(df2)
        
        self.minVal = np.minimum(min_normal,min_abnormal)
        self.maxVal = np.maximum(max_normal,max_abnormal)

    def update_hyper_param(self, hyper_param : dict):

        """ Updates the hyper param dict with Min/Max"""
        hyper_param['original_min'] = self.minVal
        hyper_param['original_max'] = self.maxVal
        return hyper_param




class DataFrameMaker():

    def __init__(self, raws_per_file : int):
        self.raws_per_file = raws_per_file
    
    def compute(self, data_list : list):
        """ Create a dataframe where each raw is considered as an experiment """
        if self.raws_per_file == 1:
            return pd.DataFrame(data_list)
        
        reshape_data_list = []
        for data_matrix in data_list:
                for data_vector in data_matrix :
                #flat_data_list.append([item for sublists in data_matrix for item in sublists])
                    reshape_data_list.append(data_vector)
        return pd.DataFrame(reshape_data_list)


def foo(normal_data : list, abnormal_data : list, hyper_param : dict):

    df_maker = DataFrameMaker(hyper_param['raws_per_file'])

    df_normal, df_abnormal = df_maker.compute(normal_data) , df_maker.compute(abnormal_data)

    norm_minmax = MinMax()

    df_normal, df_abnormal = norm_minmax.compute(df_normal,df_abnormal)

    hyper_param = norm_minmax.update_hyper_param(hyper_param)

    return df_normal,df_abnormal, hyper_param
    






def main():

    # list of lists
    list_test_1 = np.array([[1,12,4,],[1,12,4],[1,12,4]])
    list_test_2 = np.array([[[1,12],[4,64],[74,654]],[[1,12],[4,64],[74,654]]])

    hyper_param_1= {'raws_per_file' : 1,
                    'method_name' : 'psd'}
    hyper_param_2 = {'raws_per_file' : 2,
                    'method_name' : 'spectro'}



    df_normal, df_abnormal, hyper_param = foo(list_test_1, list_test_1, hyper_param_1)


    print(df_abnormal)

if __name__ == "__main__":
    main()