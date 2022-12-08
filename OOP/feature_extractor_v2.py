
from abc import abstractmethod
import librosa
import numpy as np
import system as sys

class feature_extractor():

    ### abstract methods
    @abstractmethod
    def plot(self, data_vector) -> None:
        """ Based on self.data create a plot  """
    
    @abstractmethod
    def data_processing(self, raw_data):
        """ Use raw data and process it"""
    
    @abstractmethod
    def fetch_data(self, filepath):
        """ Based on data information, fetch the appropriate data file"""

    @abstractmethod
    def write_data(self):
        """ Use processed data to write a new file"""   


class spectrogram_processing(feature_extractor):

    def __init__():
        nfft : int = 1024
        n_mels : int = 64
        power : int = 2.0
        hop_length : int = 512


    def data_processing(self, sampling_Freq : int, audio_data : float ) -> float:  # add that the output List[List]  (array of arrays)
        """ Create the mel_spectrogram from the audio file"""

        mel_spectro = librosa.feature.melspectrogram(y = audio_data,
                                                     sr = sampling_Freq,
                                                     n_fft = self.nfft,
                                                     hop_length = self.hop_length,
                                                     n_mels = self.n_mels,
                                                     power = self.power
                                                    )
        log_mel_spectro = 20.0 / self.power * np.log10(mel_spectro + sys.float_info.epsilon)
        
        return log_mel_spectro


    ### setters and getters
    def set_hyperparam(new_nfft, new_n_mels , new_power, new_hop_length):
        """ Change the hyper parameters of the spectrogram method"""                     
        nfft = new_nfft
        n_mels = new_n_mels
        power = new_power
        hop_length = new_hop_length

    
        





############################################################
############  FACTORY TO SELECT METHODS ####################

class MethodFactory(Protocol):
    def get_dataset_structure(self) -> Dataset:
        """ Return a dataset structure"""

class SpectrogramMethod():
    def get_dataset_structure(self) -> Dataset:
        return spectro_dataset

class PSDMethod():

    def get_dataset_structure(self) -> Dataset:
        return psd_dataset

def read_method(method_chosen : str) -> MethodFactory:
    factories = {'spectro': SpectrogramMethod(),
                'psd': PSDMethod()
                }
    return factories[method_chosen]


def main():
    test = feature_extractor
    print(test.hyper_parameters)

     

if __name__ == "__main__":
    
    
    method_chosen = read_method('spectro')

    main(method_chosen)