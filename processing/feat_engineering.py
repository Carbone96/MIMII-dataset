import numpy as np
import pandas as pd

from scipy import signal
import librosa

from ssqueezepy import cwt
from typing import Union



class ProcessingMethod():

    def __init__(self, hyper_param: dict = {}):
        self.processed_data = []
        self.hyper_param = hyper_param
        self.df_processed_data = None


    # setters and getters
    def get_df_processed_data(self) -> pd.DataFrame:
        return self.df_processed_data
    
    def get_hyper_param(self) -> dict:
        return self.hyper_param

    # update hyper_param informations
    def _collect_raws_per_file(self):
        """
        Returns the number of rows in the processed data.
        If the processed data is a matrix, returns the number of rows.
        If the processed data is a vector, returns 1.
        """
        if not self.processed_data:
            # the processed data variable is empty
            return 0
        if len(self.processed_data[0]) > 1:
            
            return len(self.processed_data[0])
        return 1    
 
    def update_hyper_param(self):
        self.hyper_param['raws_per_file'] = self._collect_raws_per_file()

    # store processed data as a dataframe
    def set_df_processed_data(self):
        """ Stores every 1D list or 2D list as raws in a dataframe"""
        if self.hyper_param['raws_per_file'] == 1:
            self.df_processed_data = pd.DataFrame(self.processed_data)
        
        reshape_data_list = []
        for data_matrix in self.processed_data:
                for data_vector in data_matrix :
                    reshape_data_list.append(data_vector)
        self.df_processed_data = pd.DataFrame(reshape_data_list)

    # Validation steps
    def validate_input_data(self, raw_data):

        if not isinstance(raw_data, (list, np.ndarray)):
            raise TypeError("Raw data must be a list or numpy array.")
        if isinstance(raw_data, np.ndarray) and raw_data.ndim > 2:
            raise ValueError("Raw data must be a 1D or 2D array.")

    def validate_output_data(self):

        if not isinstance(self.df_processed_data, pd.DataFrame):
            raise TypeError("Processed data must be a dataframe.")

    def compute(self):
        """ Each subclass should define its own compute method """
        pass

class ScaloClass(ProcessingMethod):

    def __init__(self, hyper_param: dict):
        super().__init__(hyper_param)

    def compute(self, raw_data):
        """ Process the data """
        for data_vector in raw_data:
            Wx, scales = cwt(data_vector, 'morlet')
            self.processed_data.append(np.array([np.mean(abs(x)) for x in Wx]))

class MFCCsClass(ProcessingMethod):

    def __init__(self, hyper_param: dict):
        super().__init__(hyper_param)

    def compute(self, raw_data):
        """ Process the data """
        fs = self.hyper_param.get('fs', 16000)
        n_mfcc = self.hyper_param.get('n_mfcc', 13)

        for data_vector in raw_data:
            mfccs = librosa.feature.mfcc(data_vector, n_mfcc = n_mfcc, sr=fs)
            mfccs = mfccs.transpose()
            self.processed_data.append(mfccs)

class PSDClass(ProcessingMethod):

    def __init__(self, hyper_param: dict):
        super().__init__(hyper_param)

    def compute(self, raw_data):
        """ Process the data """
        fs = self.hyper_param.get('fs', 16000)
        nperseg = self.hyper_param.get('nperseg', 1024)
        max_freq = self.hyper_param.get('max_freq', 3000)

        processed_data_list = []
        for data_vector in raw_data: 
            f_vec, PSDval = signal.welch(data_vector, fs ,nperseg=nperseg)
            PSD_filtered = [PSDval[f_vec < max_freq]]
            
            processed_data_list.append(PSD_filtered)
        self.processed_data = processed_data_list


class SpectroClass(ProcessingMethod):

    def __init__(self, hyper_param: dict):
        super().__init__(hyper_param)

    def compute(self, raw_data):
        """ Process the data """

        fs = self.hyper_param.get('fs', 16000) # Samplig freq
        nfft = self.hyper_param.get('nfft', 1024) # Window length
        hop_length = self.hyper_param.get('hop_length', 512) # Overlap length
        n_mels = self.hyper_param.get('n_mels', 64) # mel filters
        power = self.hyper_param.get('power', 2.0) # conversion to dB

        for data_vector in raw_data:
            mel_spectro = librosa.feature.melspectrogram(y=data_vector, sr=fs, n_fft=nfft, hop_length=hop_length, n_mels=n_mels, power=power)
            mel_spectro = mel_spectro.transpose()
            self.processed_data.append(mel_spectro)


def MethodSelector(hyper_param: dict) -> ProcessingMethod:
    """ Factory to select a processing method """

    method_dict = {
        'psd': PSDClass(hyper_param),
        'spectro': SpectroClass(hyper_param),
        'scalo': ScaloClass(hyper_param),
        'mfcc' : MFCCsClass(hyper_param)
    }

    return method_dict[hyper_param['method_name']]


def create_features(raw_data : Union[list, np.ndarray], hyper_param : dict) -> tuple[pd.DataFrame, dict]:
    """
    Processes raw data using a selected processing method and returns the processed data and hyperparameters.
    """ 
    processor = MethodSelector(hyper_param)
    processor.validate_input_data(raw_data)
    processor.compute(raw_data)
    processor.update_hyper_param()
    processor.set_df_processed_data()
    processor.validate_output_data()  
    return processor.get_df_processed_data(), processor.get_hyper_param()



def main() -> None:

    def generate_filepath(data_folder: str, machine_name: str, identifier: str, status: str) -> str:
        """ Generate the filepath in strings to get the audio file """
        return data_folder + machine_name + '/' + identifier + '/' + status

    user_path = 'C:/Users/carbo/Documents/'
    data_folder = "/MIMII/RawData/+6dB/"

    data_folder = user_path + data_folder

    identifier = 'id_02'

    hyper_param = {
        "channel": 2,
        "machine_name": 'test'
    }

    file_path = generate_filepath(data_folder=data_folder, machine_name=hyper_param['machine_name'], identifier=identifier, status="normal")

    print(file_path)

    audio_imp = AudioDataImporter(file_path=file_path, hyper_param=hyper_param, status="normal")

    hyper_param, raw_data = audio_imp._set_raw_data()

    print(f'The updated hyper_param dictionary is {audio_imp.hyper_param}')
    print(f'The raw data is a array of {len(raw_data)} lists each containing {len(raw_data[0])} elements')

    method = MethodSelector(hyper_param)
    method.foo(raw_data)
    processed_data = method.get_processed_data()
    hyper_param = method.get_hyper_param()

    print(f'The updated hyper_param dictionary is {hyper_param}')
    print(f'The processed data is a list of {len(processed_data)} lists each containing {len(processed_data[0])} elements')

if __name__ == '__main__':
    main()

