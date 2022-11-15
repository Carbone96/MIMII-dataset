import pandas as pd

class Dataset():
    
    data_pathes = ['C:/Users/carbo/Documents/MIMII/Data/+6dB/Spectrogram/', 'C:/Users/carbo/Documents/MIMII/Data/+6dB/Spectrogram_5Bands/']
    machines_names = [['fan'],['pump'],['rail'],['valve']]
    IDs_list = [['id_00'],['id_02'], ['id_04'], ['id_06']]
    def __init__(self, machine_selected, IDs_selected, method):
        self.datapath = Dataset.data_pathes[method]
        self.machines = Dataset.machines_names[machine_selected-1:machine_selected]
        self.IDs = Dataset.IDs_list[IDs_selected-1:IDs_selected]
                    
def readData(datapath):
    # Read file, remove the index raw
    # To check : do we need to remove the first raw with parquet file ? as with the CSV file 
                    df = pd.read_parquet(datapath)
                    df = df.iloc[1:]
                    return df
                        
def listToString(s):
     
    """ Convert a list to string  SHOULD BE IMPLEMENTED WITH LIST COMPREHENSION """    
        
    # initialize an empty string
    str1 = ""
    # traverse in the string
    for ele in s:
        str1 += ele
    # return string
    return str1
                  
def fun(machines_selected,IDs_selected,method):
    
    Data = Dataset(machines_selected, IDs_selected, method)
    
    df_normal = None
    df_abnormal = None
    
    
    for machine in Data.machines:
        for ID in Data.IDs:
            machine_str = listToString(machine)
            ID_str = listToString(ID)
            normal_data_name = Data.datapath + machine_str + '_' + ID_str + '_' + 'normal' + '.parquet'
            abnormal_data_name = Data.datapath + machine_str + '_' + ID_str + '_' + 'abnormal' + '.parquet'
               
            if df_normal is None:
                df_normal = readData(normal_data_name)
                df_abnormal = readData(abnormal_data_name)
            else:
                df_normal_temp = readData(normal_data_name)
                df_normal = pd.concat([df_normal,df_normal_temp],ignore_index = True)
               
                df_abnormal_temp = readData(abnormal_data_name)
                df_abnormal = pd.concat([df_abnormal,df_abnormal_temp], ignore_index = True)
                
    print('Data acquired !')
    return df_normal, df_abnormal  