import pickle
from pathlib import Path
import pandas as pd

# 1) Create a code to import a file
# 2) Separate the 3 values from the dictionnary
# 3) Perform separation on the labels on the test set

def fetch_train_test_sets(filepath : Path) -> tuple[pd.DataFrame, pd.DataFrame, int]:

    if not filepath.is_file():
        raise FileNotFoundError('file not found at specified filepath')
    print(filepath)
    data = pd.read_pickle(filepath)
  
    train_set = data['train_set']
    test_set = data['test_set']


    hyper_param = data['hyper_param']      
    labels = recreate_labels_from_hyper_param(hyper_param)        # retrieve test set labels
    test_set = test_set.drop('labels', axis=1)   # remove labels column from test set

    return train_set, test_set, labels, hyper_param['raws_per_file']

def generate_file_path(data_folder : str, machine_name : str, feat_extract_method : str, id_anomaly : str) -> Path:
    return Path(data_folder + '/' + feat_extract_method + '/' + machine_name + '/' +  id_anomaly + '.pkl')


def recreate_labels_from_hyper_param(hyper_param : dict) -> list[bool]:

    file_count_abnormal = hyper_param['file_count_abnormal']
    labels = [0] * file_count_abnormal + [1] * file_count_abnormal
    labels = pd.DataFrame(labels)
    return labels.astype(bool)     

def import_data(data_folder : str, data_settings: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:

    machine_name = data_settings['machine_name']
    feat_extract_method = data_settings['feat_extract_method']
    id_anomaly = data_settings['id_anomaly']
    filepath = generate_file_path(data_folder, machine_name, feat_extract_method, id_anomaly)
    return fetch_train_test_sets(filepath)