
import sys
import os

import numpy as np
import pandas as pd

from scipy import signal
import librosa

from abc import ABC, abstractmethod

from ssqueezepy import cwt


def MethodSelector(hyper_param : dict):
    method_dict = {'psd' : PSD_class(hyper_param),
                    'spectro' : Spectro_class(hyper_param)}

    return method_dict[hyper_param['method_name']]


class ProcessingMethod():

    def __init__(self,hyper_param :dict):
        self.processed_data = []
        self.hyper_param = hyper_param


    def update_hyper_param(self, raws_per_file: int, columns_per_file: int) -> dict:
        self.hyper_param['raws_per_file']= raws_per_file
        self.hyper_param['columns_per_file']= columns_per_file

    def get_dimensions_of_file(self):
        """ Check the type of the first value of processed file
            If it's list : the input is matrix, if not : the input is a vector"""

        if self.processed_data[0].ndim == 2:
            return np.shape(self.processed_data[0])[0], np.shape(self.processed_data[0])[1]
        return 1, len(self.processed_data)

    def get_processed_data(self):
        return self.processed_data
    
    def get_hyper_param(self):
        return self.hyper_param

    def foo(self,raw_data):
        """ Performs all method of the subclass """
        self.compute(raw_data)
        raws_per_file, columns_per_file = self.get_dimensions_of_file()
        self.update_hyper_param(raws_per_file, columns_per_file)

    @abstractmethod
    def compute(self, raw_data):
        """ Perform the computation related to a specific method"""
        pass


class Scalo_class(ProcessingMethod):

    def __init__(self,hyper_param:dict):
        super().__init__(hyper_param)

    def compute(self, raw_data):
        """ Process the data """
        for data_vector in raw_data: 
            Wx, scales = cwt(data_vector, 'morlet')
            self.processed_data.append(np.array([np.mean(abs(x)) for x in Wx]))

class PSD_class(ProcessingMethod):


    def __init__(self, hyper_param :dict):
        super().__init__(hyper_param)
        self.max_freq = hyper_param['max_freq']

    def low_pass_filter(self, PSD_list : list, f_vec : list) -> list:
        """ Create a low pass filtering of the PSD based on the max freq"""
        return PSD_list[f_vec < self.max_freq]

    def compute(self, raw_data):
        """ Process the data """
        fs = self.hyper_param['fs']
        for data_vector in raw_data: 
            f_vec, PSDval = signal.welch(data_vector, fs ,nperseg=1024)
            PSD_filtered = self.low_pass_filter(PSDval, f_vec)
            self.processed_data.append(PSD_filtered)
            
class Spectro_class(ProcessingMethod):


    def __init__(self, hyper_param :dict):
        super().__init__(hyper_param)

    def compute(self, raw_data):
        """ Process the data """
        

        """ FOR FUTURE UPDATE : 
        ----> feed in a dictionnary (i.e. hyper_param) to change the parameters of the spectrogram
        """
        nfft = 1024
        hop_length = 512
        n_mels= 64
        power = 2.0

        for data_vector in raw_data: 
            mel_spectro = librosa.feature.melspectrogram(y = data_vector,
                                                     sr = self.hyper_param['fs'],
                                                     n_fft = nfft,
                                                     hop_length = hop_length,
                                                     n_mels = n_mels,
                                                     power = power
                                                    )

            log_mel_spectro = 20.0 / power * np.log10(mel_spectro + sys.float_info.epsilon)

            self.processed_data.append(log_mel_spectro.T)
            


def foo(raw_data, hyper_param):
    
    method_dict = {'psd' : PSD_class(hyper_param),
                'spectro' : Spectro_class(hyper_param),
                'scalo' : Scalo_class(hyper_param)}

    method = method_dict[hyper_param['method_name']]
    method.foo(raw_data)
    return method.get_processed_data(), method.get_hyper_param()



def main():
    
    # Fetch some data for testing the testing process
    import importData

    user_path = 'C:/Users/carbo/Documents/'
    data_folder  = "/MIMII/RawData/+6dB/"
    FILEPATH_ABNORMAL  = "/MIMII/RawData/+6dB/test/id_02/abnormal"

    ID = 'id_02'
    hyper_param = {'channel' : 2,
                  'method_name' : 'scalo',
                  'machine_name' : 'test',
                  'max_freq' : 3000}

    audio_imp = importData.AudioDataImporter(user_path, data_folder,ID,hyper_param, status = "normal" )
    hyper_param, raw_data = audio_imp.foo()

    processed_data, hyper_param  = foo(raw_data, hyper_param)
    

    print(processed_data[0])
    print(f'{hyper_param} ')
    print(f'The dimensions of the first element of the processed data are : {np.shape(processed_data[0])}')

if __name__ == "__main__":
    main()