import tensorflow as tf
import numpy as np

def getMinMax(df):
    """ return the min & max value of a dataframe"""
    # Find minimal value in dataset
    min_val = float(tf.reduce_min(df))
    # Find maximal value in dataset
    max_val = float(tf.reduce_max(df))
    return min_val,max_val

def MinMaxSelector(df1,df2):
    
    min_normal,max_normal = getMinMax(df1)
    min_abnormal,max_abnormal = getMinMax(df2)
    
    global_min = np.minimum(min_normal,min_abnormal)
    global_max = np.minimum(max_normal,max_abnormal)

    return global_min, global_max


def MinMax_foo(df1, df2,factor=1):
    """ Get all data from an experiment and apply the same linear transformation on both dataset : dataset € [0,1] * factor """
    
    min_val, max_val = MinMaxSelector(df1, df2)
    
    df1_normalized = ((df1 - min_val) / (max_val - min_val)) * factor
    df2_normalized = ((df2 - min_val) / (max_val - min_val)) * factor
    
    return df1_normalized, df2_normalized, min_val, max_val