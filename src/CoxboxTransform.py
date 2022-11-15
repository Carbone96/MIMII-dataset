import sklearn
from sklearn.preprocessing import PowerTransformer
import sklearn.compose
import pandas as pd


def fun(df):
    
    boxcox_transform = PowerTransformer(method="box-cox")
    ct = sklearn.compose.ColumnTransformer(transformers=[['boxcox_transform',boxcox_transform,list(range(len(df.columns)))]], remainder='passthrough')
    boxcox_X = ct.fit_transform(df).copy()
    boxcox_X = pd.DataFrame(boxcox_X).copy()

    return boxcox_X

