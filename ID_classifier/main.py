import runner

"""
List of complete dictionnaries :

- dict_anomaly = {0:'id_00', 1:'id_02', 2:'id_04', 3:'id_06'}
- dict_machine = {0:'slider', 1:'fan', 2:'pump', 3:'valve'}
- dict_feat_extract_method = {0: 'psd', 1: 'spectro', 2 : 'mfcc'}

"""

DATA_FOLDER = 'C:/Users/carbo/Documents/MIMII/ProcessedData/+6dB/'

def main():

    data_settings = {'machine_name': 'fan',
                'feat_extract_method': 'spectro',
                'id_anomaly' : 'id'}

    run_settings = {'name' : 'CNN',
                    'epochs' : 1,
                    'batch_size' : 32}

    runner.run(DATA_FOLDER, run_settings, data_settings)

if __name__ == '__main__':
    main()