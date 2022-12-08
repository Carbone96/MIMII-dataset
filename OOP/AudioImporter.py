import pathlib
from abc import ABC, abstractmethod
from typing import Protocol

import os
import librosa

class AudioExporter():

    def __init__(self):
        self.filepath = ''
        self.channel = None
        self.srate = 0
        self.length = 0
        self.signal =  []

    @abstractmethod
    def prepare_export(self, audio_data):
        """ Prepares audios file to be exported"""
    @abstractmethod
    def do_export(self, folder: pathlib.Path):
        """ Export a file from a folder"""

class WAVAudioExporter(AudioExporter):
    
    """ Export WAV file using soundfile package"""
    def prepare_export(self, audio_data):
        print("Prepare export of the audio file")

    def do_export_pathfile(self, folder: pathlib.Path):
        pass

    def do_export(self, base_folder : str, file_path : str):
        self.filepath = file_path.replace(os.path.abspath(base_folder),'')
        signal,sr = librosa.load(os.path.abspath(base_folder + self.filepath), sr=None, mono=False)
        
        self.srate = sr
        self.length = len(signal[0,:])
        self.signal = signal


class ExporterFactory(Protocol):
    """ Factory that represents the different videos codecs.
    The factory does not maintain any of the instances it creates
    """
    def get_audio_exporter(self) -> AudioExporter:
        """ Return an audio exporter"""

class FastExporter():
    """ Compute fast but low quality audio file"""

    def get_audio_exporter(self) -> AudioExporter:
        pass

class HighQualityExporter():
    """ Compute at slower speed with high quality"""

    def get_audio_exporter(self) -> AudioExporter:
        pass

class MasterQualityExporter(ExporterFactory):
    """ Computing is slowest with highest audio quality"""

    def get_audio_exporter(self) -> AudioExporter:
        return WAVAudioExporter()


def read_exporter(factory_quality : str) -> ExporterFactory:
    """ Construct an audio exporter based on user preference"""
    factories = {
        'low' : FastExporter(),
        'high' : HighQualityExporter(),
        'master': MasterQualityExporter()
    }

    return factories[factory_quality]


def main(fac : ExporterFactory) -> None:
    
    audio_exporter = fac.get_audio_exporter()
    print(audio_exporter)

if __name__ == "__main__":

    fac = read_exporter('master')
    main(fac)
        