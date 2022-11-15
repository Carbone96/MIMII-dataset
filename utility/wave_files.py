
""" 
Class to contain the audio file in order to be used by different processing methods
"""

class memory_wave_files():
    def __init__(self):
        self.filepath = ''
        self.channel = None
        self.srate = 0
        self.length = 0
        
    def read_wav(self, base_folder, file_path):
        
        self.filepath = file_path.replace(os.path.abspath(base_folder),'')
        signal,sr = librosa.load(os.path.abspath(base_folder + self.filepath), sr=None, mono=False)
        
        self.srate = sr
        self.length = len(signal[0,:])
        self.channel = af
        
        return self