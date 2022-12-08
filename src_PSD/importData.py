import numpy as np
import soundfile as sf

def foo(audiofile : str, channel : int):
    try:
        multi_channel_data, sr = sf.read(audiofile)
        if multi_channel_data.ndim <= 1:
            # ADD HERE A CONDITION TO READ MULTICHANNEL LENGTH AND REMOVE IF TOO SHORT
            return sr, multi_channel_data
        return sr, np.array(multi_channel_data)[:,channel]
    except ValueError:
        pass
    
