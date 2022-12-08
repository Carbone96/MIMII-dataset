from src_PSD import importData

import sys
import numpy as np
import os
from scipy import signal
import pandas as pd
import librosa

def foo(user_path : str, data_path : str, channel : int, method_name : str):
    
    if method_name.lower() == 'psd':
        df,dic_file = PSD(user_path, data_path, channel, max_freq = 3000)
    elif method_name.lower() == 'spectro':
        df,dic_file = spectrogram(user_path, data_path, channel)
    else:
        df,dic_file = scalo(user_path, data_path)
        
    return df, dic_file

def dic_file_maker(method_list,num_files : int) -> dict:
    """ Create a dictionnary that sums up the data shape of files after processing """
    
    dic_file = { 'num_files' : num_files,
                 'raws_per_file' : method_list.shape[0],
                 'columns_per_file' : method_list.shape[1]
               }
    return dic_file


def PSD(user_path : str, data_path : str, channel : int, max_freq : int):
    """ Process the data and returns it as a dataframe """
    your_path = user_path+data_path
    files = os.listdir(your_path)
    
    PSD_list = [] 
    
    num_files = len(files)
    dic_file = None
    
    for index,file in enumerate(files):
        if os.path.isfile(os.path.join(your_path,file)):

            fs, audio = importData.foo(os.path.join(your_path,file), channel)
            f_vec, PSD = signal.welch(audio, fs, nperseg=1024)
            PSD_filtered = low_pass_filter(PSD, f_vec, max_freq)
            PSD_list.append(PSD_filtered)
    

    
    PSD_df = pd.DataFrame(PSD_list)      
    return PSD_df, dic_file

def low_pass_filter(PSD_list : list, f_vec : list, max_freq : int):
    """ Create a low pass filtering of the PSD based on the max freq"""
    return PSD_list[f_vec < max_freq]
    

    
def spectrogram(user_path :str, data_path :str, channel : int):
    """ Process the data and returns it as a dataframe """
    your_path = user_path+data_path
    files = os.listdir(your_path)
    
    log_mel_df = None
    log_mel_df_flag = False
    
    num_files = len(files)
    dic_file = None
    
    for index,file in enumerate(files):
        if os.path.isfile(os.path.join(your_path,file)):

            fs, audio = importData.foo(os.path.join(your_path,file), channel)
            
            mel_spectro = librosa.feature.melspectrogram(y = audio,
                                                     sr = fs,
                                                     n_fft = 1024,
                                                     hop_length = 512,
                                                     n_mels = 64,
                                                     power = 2.0
                                                    )
            power = 2.0
            log_mel_spectro = 20.0 / power * np.log10(mel_spectro + sys.float_info.epsilon)
            
            ### should not be in the loop, but it will for now...
            dic_file = dic_file_maker(log_mel_spectro, num_files)
            
            log_mel_df, log_mel_df_flag = concatenateDF(log_mel_df, log_mel_spectro, log_mel_df_flag)
            
            if dic_file is None:
                dic_file = dic_file_maker(PSD_list,num_files)
            
    return log_mel_df, dic_file


def concatenateDF(log_mel_df, log_mel_spectro, log_mel_df_flag):
    
    log_mel_df_temp = pd.DataFrame(log_mel_spectro)
    log_mel_df_temp.transpose()
    if log_mel_df_flag == False:

        log_mel_df = pd.concat([log_mel_df,log_mel_df_temp], axis = 0)
    else:
        log_mel_df = log_mel_df_temp
        log_mel_df_flag = True
    return log_mel_df, log_mel_df_flag
    
                                

def scalogram():
    pass