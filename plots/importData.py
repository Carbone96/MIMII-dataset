import logging
import os
import soundfile as sf
import numpy as np

from pathlib import Path
from dataclasses import dataclass, field



@dataclass
class AudioDataImporter:

    file_path : Path
    hyper_param : dict 
    status : str
    
    raw_data : list = field(init=False,default_factory=list)
    channel : int = -1
    files : list = field(init=False)
    
    def __post_init__(self):
        # Retrieve files from folder, check if they exist
        if not os.path.isdir(self.file_path):
            logging.error(f"{self.file_path} does not exist")
            raise FileNotFoundError
        self.files = list(self.file_path.glob('**/*'))
        # Check if all files in the directory are files
        if not all([f.is_file() for f in self.files]):
            logging.error(f"{self.file_path} contains folders or non-file elements")
            raise ValueError
        
        # set the channel
        self.channel = self.hyper_param['channel']

    def data_import(self, datapath : Path, min_samples = 1000):
        """ Performs the data import on a given file. Select the channel"""
        multi_channel_data, fs = sf.read(datapath)

        # In case the data is corrupted/too small 
        if len(multi_channel_data) < min_samples:
            return fs, None
        # In case only 1 channel available
        if multi_channel_data.ndim <= 1:
            return fs, multi_channel_data
        # In case multiple channel are accessible
        return fs, np.array(multi_channel_data)[:,self.channel]
        

    def _update_hyper_param(self,fs) -> dict:
        """ Update the hyper parameters dictionnary """
        file_count_str = 'file_count_' + self.status
        self.hyper_param[file_count_str] =  len(self.files)
        self.hyper_param['fs'] =  fs

    def _set_raw_data(self):
        """ Import each file located at a specific filePath (folder), import and save them in a list.
            Updates hyper-param dict
        """
        for file in self.files:
            if file.suffix != ".wav":
                    logging.warning(f"Ignoring file {file} as it is not a .wav file")
            else:
                fs, raw_data_vector = self.data_import(file)
                if raw_data_vector is not None:
                    self.raw_data.append(raw_data_vector)


        self._update_hyper_param(fs)

        return self.hyper_param, self.raw_data

def generate_filepath(data_folder : str, machine_name : str, id : str, status : str) -> Path:
        """ Generate the filepath in strings to get the audio file """
        return Path(data_folder + machine_name + '/' + id + '/' + status)

def main() -> None:

    user_path = 'C:/Users/carbo/Documents/'
    data_folder  = "/MIMII/RawData/+6dB/"

    data_folder = user_path + data_folder

    id= 'id_02'

    hyper_param = {"channel" : 2,
                    "machine_name" : 'test'}


    file_path = generate_filepath(data_folder = data_folder, machine_name= hyper_param['machine_name'], id = id , status ="normal")

    print(file_path)

    audio_imp = AudioDataImporter(file_path = file_path, hyper_param =hyper_param, status ="normal")

    hyper_param, raw_data = audio_imp._set_raw_data()
    
    print(f'The updated hyper_param dictionnary is {audio_imp.hyper_param}')
    print(f'The raw data is a array of {len(raw_data)} lists each containing {len(raw_data[0])} elements')

if __name__ == "__main__":
    main()