### Successful package import message ###
print('loading feature_extractor.py')

from dataclass import dataclass, field
from enum import Enum
from typing import List, Protocol


class processing_method_type(Enum):
    
    BASE = 0
    PSD = 1
    WELCSH_PSD = 2
    SPECTRO = 3
    SCALO = 4
    
    def select_method(method_ID : int, data_folder : str) -> feature_extractor:
        
        ID_dict = {BASE: processing_spectrogram ,
                    PSD : processing_PSD,
                   WELCSH : processing_spectro,
                   SCALO : processing_scalo}
        
        processed_data = ID_dict[method_ID](data_folder)
        
        return processed_data

    
    
@dataclass
class feature_extractor():
    
    def __init__(self):
           
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
        data : list = field(default_factory = [])

        
    ### abstract methods
    
        
    
    
    
        
    ### setters and getters    
        
    def set_hyper_parameters_copy(self, ft_extractor : feature_extractor) -> self:
        """ Copies hyper parameters from another feature_extractor """
        self.hyper_parameters = ft_extractor.hyper_parameters
        
    def get_data(self):
        """ Returns processed data"""
        return self.data
        
    def _get_complete_filepath(self, filepath = None) -> str:
        
        if filepath:
            return os.abspath(self.data_filepath + filepath)
        else 
        
        return complete_filepath
    
        
        
                         
                        