print('loading mel_spectrogram.py')

# Feature extractor for handling mel spectrogram

class feature_extractor_spectrogram( feature_extractor ):
    
    def __init__(self, base_folder, name = 'mel_spectro'):
        super().__init__(base_folder,
                         name,
                         xlabel='time',
                         ylabel='freq',
                         zlabel='log mel energy'
                        )
        # set type
        self.para_dict['type'] = feature_extractor_type.MEL_SPECTRUM
        self.para_dict['type_name'] = 'MEL'
        
        # default hyper
        self.set_hyperparameters()
        
    def set_hyperparameters(self,
                                n_mels = 64,
                                nfft = 1024,
                                power = 2.0,
                                hop_length = 512,
                                channel = 0
                               ):
        
        """ Initialize the parameters to use for the spectrogram """
        
        self.para_dict['hyperpara'] = { \
                                       'n_mels' : n_mels,
                                       'nfft' : nfft,
                                       'power' : power,
                                       'hop_length' : hop_length,
                                       'channel' : channel}
        
        self.para_dict['file_nameHyperParameter_string'] = 'nm'  + str(n_mels)
        
        if os.path.isfile( self._full_wave_path() ):
            #print('recalc mel')
            self.create_from_wav(self.para_dict['wave_filepath'])
            
    def create_from_wav(self, filepath):
        
        """ Import the soundfile, read it and create the mel spectrogram"""
        
        # calc librosa
        
        channel = self.para_dict['hyperpara']['channel']
        self.para_dict['data_channel_use_str'] = 'ch' + str(channel)
        
        self.para_dict['wave channel'] = [channel]
        signal = np.array(self._read_wav(filepath))[channel,:]
        power = self.para_dict['hyperpara']['power']
        
        mel_spectro = librosa.feature.melspectrogram(y = signal,
                                                     sr = self.para_dict['wave_srate'],
                                                     n_fft = self.para_dict['hyperpara']['n_fft'],
                                                     hop_length = self.para_dict['hyperpara']['hop_length'],
                                                     n_mels = self.para_dict['hyperpara']['n_mels'],
                                                     power = self.para_dict['hyperpara']['power']
                                                    )
        log_mel_spectro = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)
        
        self.feature_data = log_mel_spectro
        
    
    def plot(self, colorbarFlag = True):
        
        """ Plot the melspectrogram """
        
        librosa.display.specshow(self.feature_data,
                                 x_axis = 'time [s]',
                                 y_axis = 'freq [Hz]',
                                 sr = self.para_dict['wave_srate'])
        plt.title('Mel spectrogram' + self.para_dict['wave_pathfile'])
        if colorbarFlag:
            plt.colorbar(format = '%+2.0f dB')
    
    
    def time_axis(self):
        """ Return the timevector of the soundfile """
        
        return np.linspace(0, self.para_dict['wave_length'] / self.para_dict['srate'], len(self.feature_data[:,0]))
    
    def freq_axis(self):
        """ Return the frequency vector of the soundfile """
        
        return np.linspace(0, self.para_dict['srate']/2,  self.para_dict['hyperpara']['n_mels'])
    
    
    def frame_feature(self, frames):
        
        """ Create the same input feature vector as in the MIMII paper """
        
        vectorarray_size = len(self.feature_data[0, :]) - frames + 1
        n_mels = self.feature_data.shape[0]
        
        vectorarray = np.zeros((vectorarray_size, n_mels * frames ), float)
        for frame in range(frames):
            vectorarray[:, n_mels * frame : n_mels * (frame +1)] = self.feature_data[:, frame : frame + vectorarray_size].T
            
        return vectorarray
    
    
    def get_feature(self, feat_para_dict):
        
        """ Select which type of input vector to generate : simple 'flattening' or 'frames' as in MIMII paper """
        
        if feat_para_dict['function'] == 'flat':
            return self.flat_feature()
        elif feat_para_dict['function'] == 'frame':
            return self.frame_feature(feat_para_dict['frames'])
        else:
            raise Exception("ValueError : function " + feat_para_dict['function'] + " is unknown !")
            
    def flat_feature(self):
        return self.feature_data.flatten()
    