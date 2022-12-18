import processing_methods
import preparation
import compute_metrics
import splitter
import evaluation
import learner
import importData

import numpy as np

CHANNELS = {"fan" : 5,"pump": 3,"valve" : 1, "slider" : 7, "test" : 5} # il faudrait vérifier que je sélectionne les bons channels, pour le moment ça fera l'affaire
IDS = [["id_00"],["id_02"], ["id_04"], ["id_06"]]
EPOCHS = 50
BATCH = 512

"""
        OVERALL UPDATE IMPLEMENTATION :
            ---> Ideally, they're should be a 'settings' dict with overall methods settings and a 'hyper_param' dict

"""


def foo( data_folder : str, hyper_param :dict):
    
    IDs_chosen = IDS[hyper_param['IDchosen']]
    hyper_param['channel'] = CHANNELS[hyper_param['machine_name']]
    AUCs = []

    for ID in IDs_chosen:
        
        
        """ 
        UPDATE TO DO : 
            The way hyper_param is updated by importData is not the best !  

            ---> Ideally, they're should be a 'settings' dict with overall methods settings and a 'hyper_param' dict

        """
        file_path_normal = importData.generate_filepath(data_folder, hyper_param['machine_name'], ID, "normal")
        file_path_abnormal = importData.generate_filepath(data_folder, hyper_param['machine_name'], ID, "abnormal")

        importer_RawData_normal = importData.AudioDataImporter(file_path= file_path_normal,hyper_param = hyper_param, status = "normal")
        hyper_param, raw_data_normal = importer_RawData_normal._set_raw_data()

        importer_RawData_abnormal = importData.AudioDataImporter(file_path= file_path_abnormal,hyper_param = hyper_param, status = "abnormal")
        hyper_param, raw_data_abnormal = importer_RawData_abnormal._set_raw_data()

        machine_name = hyper_param['machine_name']
        print(f'Data import of {machine_name} with {ID} has been successful !')


        """
        UPDATE TO DO :

        """
        processed_data_normal, hyper_param = processing_methods.foo(raw_data_normal, hyper_param)
        processed_data_abnormal, hyper_param = processing_methods.foo(raw_data_abnormal, hyper_param)

        print(f'Data processing has been successful !')
        # Prepare the data before splitting it into train & test sets 
        df_normal, df_abnormal, hyper_param = preparation.foo(processed_data_normal, processed_data_abnormal, hyper_param)
        # Create the test & trains sets (not randomized !)

        """
        UPDATE TO DO:
        
        """
        # ici on laisse bien le train & test sets avec leurs labels !
        train_set, test_set = splitter.foo(df_normal,df_abnormal,hyper_param)
        print(f'Data preparation is finished.. beginning learning !')

        """
        UPDATE TO DO:

        Only 1 learner available : 
        ---> Return the learner trained instead of an instance
        ---> Allow to change the settings of the learner
        ---> Instanciate the learner in order to use a Factory
        ---> Feed in a 'learner_method' from hyper_param to choose a learner methodology

        WARNING :
            POOR FLEXIBILITY :
        ---> I don't like that the 'runner.py' as to run the autoencoder.fit() , other function might not be implementable this way !
        """
        # Train algorithm 
        autoencoder = learner.createLearner(train_set,test_set,hyper_param, latent_dim = hyper_param['latent_dim'], loss_fun= "mse")
       
        # Test algorithm
        print(f'Ready to evaluate performances !')
        lossValues, labels = evaluation.foo(autoencoder, test_set, hyper_param)
        AUCs.append(compute_metrics.get_AUC_score(labels, lossValues))
        print(f'AUC value stored !')


        avg_normal_MSE = np.mean(lossValues[labels])
        avg_abnormal_MSE = np.mean(lossValues[~labels])

    return AUCs, avg_abnormal_MSE, avg_normal_MSE
    


    