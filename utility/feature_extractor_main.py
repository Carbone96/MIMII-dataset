print('loading feature_extractor_main.py')


# Enumeration of the differents IDs possible for feature extraction of the soundfile
from enum import Enum
class feature_extractor_type(Enum):
    BASE = 0
    PSD = 1
    MEL_SPEC = 2
    WELCSH_PSD = 3
    SCALO = 4
    
    def feature_extractor_from_dict(dic, base_folder):
        if dic['para_dict']['type'] = feature_extractor_type.MEL_SPEC:
            feat = feature_extractor_spectrogram(base_folder)
            feat.read_from_dict(dic)

        elif dic['para_dict']['type'] = feature_extractor_type.PSD:
            feat = feature_extractor_PSD(base_folder)
            feat.read_from_dict(dic)

        elif dic['para_dict']['type'] = feature_extractor_type.WELCSH_PSD:
            feat = feature_extractor_welcshPSD(base_folder)
            feat.read_from_dict(dic)

        elif dic['para_dict']['type'] = feature_extractor_type.SCALO:
            feat = feature_extractor_scalo(base_folder)
            feat.read_from_dict(dic)

        return feat

    def feature_extractor_from_file():
        d = pickle.load(open(filepath, "rb"))
        return feature_extractor_from_dict(d, base_folder)

# Main class : works as an API
class feature_extractor():
    def __init__(self, base_folder, name = 'base_feature', xlabel='x', ylabel='y', zlabel='z'):
        
        self.para_dict={'name' : name,
                        'xlabel' : xlabel,
                        'ylabel': ylabel,
                        'zlabel' : zlabel,
                        'type_name' : 'BASE',
                        'data_channel_use_str' : '',
                        'type' : feature_extractor_type.BASE,
                        'file_name_mainHyperParameter_String' : '',
                        'wave_filepath' : '',
                        'wave_str' : 0,
                        'wave_length' : 0,
                        'wave_channel' : 0,
                        'hyperPara' : {}
                       }
        self.base_folder = base_folder
        self.feature_data = None
        
        ###  setters and getters
        
        @property
        def name(self):
            return self.para_dict['name']
                        
        @property
        def file_name_mainHyperParameter_String(self):
            return self.para_dict['file_name_mainHyperParameter_String']
                        
        @property
        def type_str(self):
            return self.para_dict['type_name']
        
        @property
        def data_channel_use_str(self):
            return self.para_dict['data_channel_use_str']

        
    
    
        def set_hyperparamter_from_fe(self,fe):
            self.para_dict['hyperpara'] = fe.para_dict['hyperpara']
            
        
        def get_dict(self):
        return {'para_dict': self.para_dict,
                'feature_data': self.feature_data}
        
        
        
        ### private functions
        
        def _str_(self):
            return '<'+str(self.para_dict['type']) + '>[' + str(self.para_dict['hyperpara']) + ']' + 'wav=' +str(self._full_wave_path())
        
        def _full_wave_path(self, filepath=None):
            if filepath:
                return os.path.abspath(self.base_folder + filepath)
            else:
                return os.path.abspath(self.base_folder + self.para_dict['wave_filepath'])
        
        def _read_wav(self, filepath):
            # if type(filepath) is str:
            filepath = filepath.replace(os.path.abspath(self.base_folder),'')
            self.para_dict['wave_filepath'] = filepath
            signal, sr = librosa.load(self._full_wave_path(filepath), sr=None, mono=False)
            self.para_dict['wave_srate'] = sr
            self.para_dict['wave_length'] = len(signal[0])
            return signal
                
         
        ### import data/read data
        
        def read_from_dict(self, d):
            self.para_dict = d['para_dict']
            self.feature_data = d['feature_data']
            
        def save_to_file(self, filepath):
            pickle.dump(self.get_dict(),
                    open( filepath, "wb" ) )
            # TODO catch errors and handling
       
        