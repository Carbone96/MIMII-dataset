
import feat_engineering
import normalization
import splitter
import data_importer
import save_sets


channels = {"fan" : 5,"pump": 3,"valve" : 1, "slider" : 7, "test" : 5} 


def process(data_folder : str, write_folder : str, hyper_param :dict, saving : bool = False):

    id_anomaly = hyper_param['anomaly_chosen']
    hyper_param['channel'] = channels[hyper_param['machine_name']]
    machine_name = hyper_param['machine_name']
    
    raw_data_normal, raw_data_abnormal, hyper_param = data_importer.importer(data_folder, machine_name, id_anomaly, hyper_param)
    print(f'Data import of {machine_name} with {id_anomaly} has been successful !')

    df_normal, hyper_param = feat_engineering.create_features(raw_data_normal, hyper_param)
    df_abnormal, hyper_param = feat_engineering.create_features(raw_data_abnormal, hyper_param)

    df_normal_normalized, df_abnormal_normalized, hyper_param = normalization.normalize(df_normal, df_abnormal, hyper_param)


    # ici on laisse bien le train & test sets avec leurs labels !
    train_set, test_set = splitter.split(df_normal_normalized,df_abnormal_normalized,hyper_param)
    print(f'Data processing has been successful !')

    # Normalize data : perform minmax-normalization 

    """
    UPDATE TO DO on NORMLIZATION

        1° It may be useful to pass the 'Processing method' as parameter to perform specific normalization
        2° Implementing Cox-Box transform might be a good idea on some datasets
    """

    # Create the test & trains sets (not randomized !)
    print(f'Data preparation is finished! ..saving it to designed folder !')
    if saving:
        save_sets.saving(write_folder, train_set, test_set, hyper_param)

