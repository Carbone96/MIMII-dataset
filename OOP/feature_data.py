print('loading feature_extractor.py')

from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Protocol
import os


############################################################
############  DATACLASSES INITIALIZATION ###################
    
@dataclass
class Dataset():
    
    """ Mother class for dataset structures : initialize all relevant parameters to allow for one-file at a time processing"""

    name : str = 'nameless'
    xlabel : str = 'x'
    ylabel : str = 'y'
    zlabel : str = 'z'
    method_name : str = 'BASE'
    method_val : int = 0
    channel_name : str = ''
    channel_val : int = 0
    base_folder : str = ''
    data_filepath : str = None
    data_length : int = 0
    data_vector : list = field(default_factory = [])

    ### define which methods is used

    def set_spectrogram(self):
        self.name = 'mel_spectro'
        self.xlabel = 'time'
        self.ylabel = 'freq'
        self.zlabel = 'log mel energy'
    
    def set_psd(self):
        self.name = 'psd'
        self.xlabel = 'freq'
        self.ylabel = 'amplitude'


    ### setters and getters    
        
    def set_hyper_parameters_copy(self, ft_extractor):
        """ Copies hyper parameters from another feature_extractor """
        self.hyper_parameters = ft_extractor.hyper_parameters
        
    def get_data(self):
        """ Returns processed data"""
        return self.data_vector
        
    def _get_complete_filepath(self):
        if self.data_filepath:
            return os.abspath(self.base_folder + self.data_filepath)
        else:
            print(f'Filepath missing')

    def set_channel(self, channel_in_use : int):
        """ Define which channel should be in use (depend on the machine to use)"""
        self.channel = channel_in_use




#################################################################
############  MAIN FUNCTION FOR TEST PURPOSES ###################

def main() -> None:

    pass


if __name__ == "__main__":

   pass