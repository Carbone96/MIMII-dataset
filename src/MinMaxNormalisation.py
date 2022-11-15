import tensorflow as tf

def fun(df,factor=1):
        # Find minimal value in dataset
    min_val = float(tf.reduce_min(df))
    # Find maximal value in dataset
    max_val = float(tf.reduce_max(df))
    return ((df - min_val) / (max_val - min_val)) * factor