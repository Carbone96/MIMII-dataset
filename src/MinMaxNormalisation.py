import tensorflow as tf

def getMinMax(df):
    # Find minimal value in dataset
    min_val = float(tf.reduce_min(df))
    # Find maximal value in dataset
    max_val = float(tf.reduce_max(df))
    return min_val,max_val

def fun(df,min_val,max_val,factor=1):
    
    return ((df - min_val) / (max_val - min_val)) * factor