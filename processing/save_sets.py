import pickle
from pathlib import Path
import pandas as pd

def saving(write_folder : str,  train_set : pd.DataFrame, test_set : pd.DataFrame, hyper_param : dict):

    filepath = Path(write_folder + hyper_param['machine_name'] + '/' + hyper_param['anomaly_chosen'] + '.pkl')
    exp_dict = {'train_set' : train_set, 'test_set' : test_set, 'hyper_param' : hyper_param}
    # create a binary pickle file 
    with open(filepath, 'wb') as f:
        # write the python object (dict) to pickle file
        pickle.dump(exp_dict, f)
    # close file
    f.close()


def main():
    pass

if __name__ == "__main__":

    main()

