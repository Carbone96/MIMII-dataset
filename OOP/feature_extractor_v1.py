print('loading feature_extractor.py')

from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Protocol
import os

class processing_method_type(Enum):
    
    BASE = 0
    PSD = 1
    WELCSH_PSD = 2
    SPECTRO = 3
    SCALO = 4
    
    def select_method(method_ID : int, data_folder : str):
        
        ID_dict = {BASE: processing_spectrogram ,
                    PSD : processing_PSD,
                   WELCSH : processing_spectro,
                   SCALO : processing_scalo}
        
        processed_data = ID_dict[method_ID](data_folder)
        
        return processed_data

    
    
@dataclass
class feature_extractor():
    
    name : str = 'nameless'
    xlabel : str = 'x'
    ylabel : str = 'y'
    zlabel : str = 'z'
    method_name : str = 'BASE'
    method_val : int = processing_method_type.BASE
    channel_name : str = ''
    channel_val : int = 0
    base_folder : str = ''
    data_filepath : str = ''
    data_length : int = 0
    hyper_parameters : dict = field(default_factory = {})
    data_vector : list = field(default_factory = [])

        
    ### abstract methods
    @abstractmethod
    def plot(data_vector) -> None:
        """ Based on self.data create a plot  """
    
    @abstractmethod
    def data_processing(self, raw_data):
        """ Use raw data and process it"""
    
    @abstractmethod
    def fetch_data(self):
        """ Based on data information, fetch the appropriate data file"""

    @abstractmethod
    def write_data(self):
        """ Use processed data to write a new file"""   

    ### setters and getters    
        
    def set_hyper_parameters_copy(self, ft_extractor):
        """ Copies hyper parameters from another feature_extractor """
        self.hyper_parameters = ft_extractor.hyper_parameters
        
    def get_data(self):
        """ Returns processed data"""
        return self.data
        
    def _get_complete_filepath(self):
        if self.data_filepath:
            return os.abspath(self.base_folder + self.data_filepath)
        else:
            print(f'Filepath missing')
    
        
def main():

    print(type(feature_extractor))
    testVal = feature_extractor
    print(testVal.xlabel)


if __name__ == "__main__":
    main()