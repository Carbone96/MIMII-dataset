import runner
import pandas as pd

"""
List of complete dictionnaries :

- dict_anomaly = {0:'id_00', 1:'id_02', 2:'id_04', 3:'id_06'}
- dict_machine = {0:'slider', 1:'fan', 2:'pump', 3:'valve'}
- dict_feat_extract_method = {0: 'psd', 1: 'spectro', 2 : 'mfcc'}


"""

DATA_FOLDER = 'C:/Users/carbo/Documents/MIMII/ProcessedData/+6dB/'


def run_all_machines(data_settings, run_settings, dict_anomaly, dict_machine, dict_feat_extract_method):

    AUC_list = []
    for index,method in dict_feat_extract_method.items():
        data_settings['feat_extract_method'] = method                               #select processing method
        for index,machine_name in dict_machine.items():
            data_settings['machine_name'] = machine_name                            #select machine name
            for index,anomaly_id in dict_anomaly.items():
                data_settings['id_anomaly'] = anomaly_id                            #select id
                AUC_list.append( runner.run(DATA_FOLDER, run_settings, data_settings))

    return AUC_list


def save_AUC_as_csv(AUC_list,run_settings):

    AUC_list_aggregated_by_machine = []
    for i in range(4):
        AUC_list_aggregated_by_machine.append([AUC_list[i*4 +k] for k in range(4)])

    df = pd.DataFrame(AUC_list_aggregated_by_machine, index = ['slider', 'fan', 'pump', 'valve'],columns = ['id_00','id_02','id_04','id_06'])


    df_name = 'spectro_mahala_AEsparse' + str(run_settings['latent_dim']) + 'MSE.csv'
    df.to_csv(df_name)

def main():
    data_settings = {'machine_name': 'machine',
                'feat_extract_method': 'method',
                'id_anomaly' : 'id'}


    dict_anomaly = {0:'id_00', 1:'id_02', 2:'id_04', 3:'id_06'}
    dict_machine = {0:'slider', 1:'fan', 2:'pump', 3:'valve'}
    dict_feat_extract_method = {0:'spectro'}

    latent_dim_list = [6]
    for latent_dim in latent_dim_list:
        latent_dim = latent_dim +6

        run_settings = {
                    'loss_fun' : 'mse',
                    'latent_dim' : latent_dim,
                    'name' : 'autoencoder_sparse',
                    'epochs' : 50,
                    'batch_size' : 32,
                    'evaluation_method' : 'mahalanobis'}

        AUCs = run_all_machines(data_settings, run_settings, dict_anomaly, dict_machine, dict_feat_extract_method)
        
        save_AUC_as_csv(AUCs,run_settings)



if __name__ == '__main__':
    main()