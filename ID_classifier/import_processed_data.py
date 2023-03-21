import pickle
from pathlib import Path
import pandas as pd
import os
import numpy as np
# 1) Create a code to import a file
# 2) Separate the 3 values from the dictionnary
# 3) Perform separation on the labels on the test set


################################ THIS PART IS FOR GETTING THE DATA ################################
def generate_file_path(data_folder: str, machine_name: str, feat_extract_method: str, id_anomaly: str) -> Path:
    return Path(os.path.join(data_folder, feat_extract_method, machine_name, f"{id_anomaly}.pkl"))


def create_labels_from_hyper_param(hyper_param : dict) -> pd.DataFrame:
    train_labels = [0] * (hyper_param['file_count_normal'] - hyper_param['file_count_abnormal'])
    test_labels = [0] * hyper_param['file_count_abnormal'] + [1] * hyper_param['file_count_abnormal']
    return pd.DataFrame(train_labels, dtype=bool), pd.DataFrame(test_labels, dtype=bool)


def fetch_train_test_sets(filepath : Path) -> tuple[pd.DataFrame, pd.DataFrame, int]:

    if not filepath.is_file():
        raise FileNotFoundError('file not found at specified filepath')
    data = pd.read_pickle(filepath)
  
    train_set = data['train_set']
    test_set = data['test_set']

    hyper_param = data['hyper_param']      
    train_labels, test_labels = create_labels_from_hyper_param(hyper_param)   # retrieve test set labels

    test_set = test_set.drop('labels', axis=1)   # remove labels column from test set

    return train_set, test_set, train_labels,test_labels, hyper_param



################################ THIS PART IS FOR FORMATTING THE DATA ################################

def reassign_labels(labels: pd.DataFrame, id_val: int) -> pd.DataFrame:
    return pd.DataFrame(np.where(labels, -1, id_val), columns=labels.columns)


def import_data(data_folder : str, data_settings: dict) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame,int]:

    machine_name = data_settings['machine_name']
    feat_extract_method = data_settings['feat_extract_method']

    anomaly_ids = ['id_00', 'id_02', 'id_04', 'id_06']

    train_set = pd.DataFrame()
    test_set = pd.DataFrame()
    train_labels = pd.DataFrame()
    test_labels = pd.DataFrame()
    raws_per_file = 0

    for index,anomaly_ID in enumerate(anomaly_ids):
        filepath = generate_file_path(data_folder, machine_name, feat_extract_method, anomaly_ID)
        train_set_temp, test_set_temp,train_labels_temp, test_labels_temp, raws_per_file = fetch_train_test_sets(filepath)

        train_labels_temp = reassign_labels(train_labels_temp, index)
        test_labels_temp = reassign_labels(test_labels_temp, index)

        train_set = pd.concat([train_set, train_set_temp])
        test_set = pd.concat([test_set, test_set_temp])
        train_labels = pd.concat([train_labels, train_labels_temp])
        test_labels = pd.concat([test_labels, test_labels_temp])

        
    return train_set, test_set, train_labels, test_labels, raws_per_file