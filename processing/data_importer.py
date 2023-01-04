import logging
import soundfile as sf
import numpy as np

from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class AudioDataImporter:

    file_path: Path
    hyper_param: dict 
    status: str
    channel: int = 0
    fs : int = 0

    raw_data: list = field(init=False, default_factory=list)
    files: list = field(init=False)
    

    @property
    def get_raw_data(self):
        return self.raw_data

    @property
    def get_hyper_param(self):
        return self.hyper_param


    def __post_init__(self):
        # Retrieve files from folder, check if they exist
        if not self.file_path.is_dir():
            logging.error(f"{self.file_path} does not exist")
            raise FileNotFoundError()
        self.files = list(self.file_path.glob('**/*'))
        # Check if all files in the directory are files
        if not all([f.is_file() for f in self.files]):
            logging.error(f"{self.file_path} contains folders or non-file elements")
            raise ValueError()
        # set the channel
        self.channel = self.hyper_param['channel']

    def data_import(self, data_path: Path, min_samples=1000):
        """ Performs the data import on a given file. Select the channel"""
        multi_channel_data, fs = sf.read(data_path)

        # In case the data is corrupted/too small 
        if len(multi_channel_data) < min_samples:
            return fs, None
        # In case only 1 channel available
        if multi_channel_data.ndim <= 1:
            return fs, multi_channel_data
        # In case multiple channels are accessible
        return fs, np.array(multi_channel_data)[:, self.channel]
        

    def _update_hyper_param(self) -> dict:
        """ Update the hyper parameters dictionary """
        file_count_str = 'file_count_' + self.status
        self.hyper_param[file_count_str] = len(self.files)
        self.hyper_param['fs'] = self.fs

    def _collect_raw_data(self):
        """ Import each file located at a specific filePath (folder), import and save them in a list.
            Updates hyper-param dict
        """
        for file in self.files:
            if file.suffix != ".wav":
                    logging.warning(f"Ignoring file {file} as it is not a .wav file")
            fs, raw_data_vector = self.data_import(file)
            if raw_data_vector is not None:
                self.raw_data.append(raw_data_vector)

        self.fs = fs


def generate_filepath(data_folder: str, machine_name: str, id: str, status: str) -> Path:
        """ Generate the filepath in strings to get the audio file """
        return Path(data_folder + machine_name + '/' + id + '/' + status)


def importer(data_folder : str, machine_name : str, id_anomaly : str, hyper_param : dict) -> tuple([list, list, dict]):
    """
    This function imports audio data for both normal and abnormal conditions for a given machine.
    
    Parameters:
    data_folder (str): The path to the folder containing the audio data files.
    machine_name (str): The name of the machine.
    ID (str): The ID of the machine.
    hyper_param (dict): The hyper-parameters of the processing method.
    
    Returns:
    tuple: A tuple containing the raw data for the normal and abnormal conditions, and the updated hyperparameters.
    """

    normal_path = generate_filepath(data_folder, machine_name, id_anomaly, "normal")
    abnormal_path = generate_filepath(data_folder, machine_name, id_anomaly, "abnormal")

    importer_RawData_normal = AudioDataImporter(file_path= normal_path,hyper_param = hyper_param, status = "normal")
    importer_RawData_normal._collect_raw_data()
    importer_RawData_normal._update_hyper_param()

    # Retrieve the updated dict to be fed to the next importer
    hyper_param = importer_RawData_normal.get_hyper_param

    importer_RawData_abnormal = AudioDataImporter(file_path= abnormal_path,hyper_param = hyper_param, status = "abnormal")
    importer_RawData_abnormal._collect_raw_data()
    importer_RawData_abnormal._update_hyper_param()

    return importer_RawData_normal.get_raw_data, importer_RawData_abnormal.get_raw_data, importer_RawData_abnormal.get_hyper_param



def main() -> None:
    hyper_param = {'channel' : 0}
    data_folder = 'C:/Users/carbo/Documents/MIMII/RawData/+6dB/'
    machine_name = 'test'
    id_anomaly = 'id_00'


    normal_path = generate_filepath(data_folder, machine_name, id_anomaly, "normal")

    importer_RawData_normal = AudioDataImporter(file_path= normal_path,hyper_param = hyper_param, status = "normal")


    importer_RawData_normal._collect_raw_data()
    importer_RawData_normal._update_hyper_param()

    print(importer_RawData_normal.hyper_param)



if __name__ == "__main__":
    main()