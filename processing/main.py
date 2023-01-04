import data_processing

data_folder = 'C:/Users/carbo/Documents/MIMII/RawData/+6dB/'
write_path = 'C:/Users/carbo/Documents/MIMII/ProcessedData/+6dB/'

def main():

    hyper_param = {}

    dict_anomaly = {0:'id_00'}
    dict_machine = {0:'slider', 1:'fan', 2:'pump', 3:'valve'}
    #dict_machine = {0 : 'test'}
    dict_method = {1:'spectro'}

    for index,method in dict_method.items():
        hyper_param['method_name'] = method                             #select processing method
        write_folder = write_path + method + '/'                       #select proper writing folder
        for index,machine_num in dict_machine.items():
            hyper_param['machine_name'] = machine_num                   #select machine name
            for index,anomaly_id in dict_anomaly.items():
                hyper_param['anomaly_chosen'] = anomaly_id              #select id
         
                data_processing.process(data_folder, write_folder, hyper_param, saving = False) # process & save

if __name__ == '__main__':
    main()