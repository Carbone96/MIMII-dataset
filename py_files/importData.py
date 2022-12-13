import os


"""
Message from 09/12
---------------------------------- WARNING -----------------------------------------------------------

UPDATE REQUIRED :
- It would be better to use the library FILEPATH instead of dealing with string. 
-----> Filepath can managed automatically the proper format to create for having filepath in Windows !


------------------------------------------------------------------------------------------------------
"""



class AudioDataImporter():

    def __init__(self, user_path : str, data_path : str, ID :str, hyper_param : dict, status :str):
        self.files_path = user_path + generate_filepath(data_path,ID,status,hyper_param['machine_name'])
        self.files = None
        self.file_count = None
        self.raw_data = []
        self.fs = 0
        self.channel = hyper_param['channel']
        self.hyper_param = hyper_param
        self.status = status
    # setters and getters   
    def set_files(self):
        self.files = os.listdir(self.files_path)

    # update dictionnary of hyper parameters
    def update_hyper_param(self,fs,status) -> dict:
        """ Update the hyper parameters dictionnary """
        file_count_str = 'file_count_' + self.status
        self.hyper_param[file_count_str] =  len(self.files)
        self.hyper_param['fs'] =  fs

    def VerifyFileExist(self,file):
        """ Checks that the file exists"""
        return os.path.isfile(os.path.join(self.files_path, file))

    
    def data_import(self, data_path):
        """ Performs the data import on a given file. Select the channel"""
        try:
            multi_channel_data, fs = sf.read(data_path)
            if multi_channel_data.ndim <= 1:
                # ADD HERE A CONDITION TO READ MULTICHANNEL LENGTH AND REMOVE IF TOO SHORT
                return fs, multi_channel_data
            return fs, np.array(multi_channel_data)[:,self.channel]
        except ValueError:
            pass

    def foo(self):
        """ Import each file located at a specific filePath (folder), import and save them in a list.
            Updates hyper-param dict
        """
        self.set_files()
        for file in self.files:
            if self.VerifyFileExist(file):

                fs, raw_data_vector = self.data_import(os.path.join(self.files_path,file))
                self.raw_data.append(raw_data_vector)

        self.update_hyper_param(fs, self.status)

        return self.hyper_param, self.raw_data

def generate_filepath(base_folder, ID ,status , machine_name ) :
    """ Generate the filepath to get the audio file """
    
    return base_folder + machine_name + '/' + ID + '/' + status


def generate_filepath(base_folder : str, ID : int,status : str, machine_name : str) -> str:
    """ Generate the filepath to get the audio file """
    
    return base_folder + machine_name + '/' + ID + '/' + status

    


def main() -> None:

    user_path = 'C:/Users/carbo/Documents/'
    DATAPATH_NORMAL  = "/MIMII/RawData/+6dB/"
    ID= 'id_02'

    FILEPATH_ABNORMAL  = "/MIMII/RawData/+6dB/test/id_02/abnormal"
    hyper_param = {"channel" : 2,
                    "machine_name" : 'test'}

    audio_imp = AudioDataImporter(user_path, DATAPATH_NORMAL,ID,hyper_param , "normal")
    hyper_param, raw_data = audio_imp.foo()
    
    print(f'The updated hyper_param dictionnary is {audio_imp.hyper_param}')
    print(f'The raw data is a array of {len(raw_data)} lists each containing {len(raw_data[0])} elements')

if __name__ == "__main__":
    main()