import processing_methods
import preparation
import compute_metrics
import splitter
import evaluation
import learner
import importData

CHANNELS = {"fan" : 5,"pump": 3,"valve" : 1, "slider" : 7, "test" : 5} # il faudrait vérifier que je sélectionne les bons channels, pour le moment ça fera l'affaire
IDS = [["id_00"],["id_02"], ["id_04"], ["id_06"]]
EPOCHS = 50
BATCH = 512

"""
        OVERALL UPDATE IMPLEMENTATION :
            ---> Ideally, they're should be a 'settings' dict with overall methods settings and a 'hyper_param' dict

"""


def foo( user_path : str, data_folder : str, hyper_param :dict):
    
    IDs_chosen = IDS[hyper_param['IDchosen']]
    hyper_param['channel'] = CHANNELS[hyper_param['machine_name']]
    AUCs = []

    for ID in IDs_chosen:
        
        
        """ 
        UPDATE TO DO : 
            The way hyper_param is updated by importData is not the best !  

            ---> Ideally, they're should be a 'settings' dict with overall methods settings and a 'hyper_param' dict

        """
        importer_RawData_normal = importData.AudioDataImporter(user_path,data_folder,ID,hyper_param, status = "normal")
        importer_RawData_abnormal = importData.AudioDataImporter(user_path,data_folder,ID,hyper_param, status = "abnormal")

        hyper_param, raw_data_normal = importer_RawData_normal.foo()
        hyper_param, raw_data_abnormal = importer_RawData_abnormal.foo()

        print(f'Data import has been successful !')


        """
        UPDATE TO DO :

        """
        processed_data_normal, hyper_param = processing_methods.foo(raw_data_normal, hyper_param)
        processed_data_abnormal, hyper_param = processing_methods.foo(raw_data_abnormal, hyper_param)

        print(f'Data processing haas been successful !')
        # Prepare the data before splitting it into train & test sets 
        df_normal, df_abnormal, hyper_param = preparation.foo(processed_data_normal, processed_data_abnormal, hyper_param)

        # Create the test & trains sets (not randomized !)

        """
        UPDATE TO DO:
        
        """

        # ici on laisse bien le train & test sets avec leurs labels !
        train_set, test_set = splitter.foo(df_normal,df_abnormal,hyper_param)
        print(f'Data preparation is finished.. beginning learning !')
        print(type(train_set))
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
        print(f'Shape of train set : {train_set.shape}')
        print(f'Shape of train set : {test_set.shape}')


        # Train algorithm 
        autoencoder = learner.foo(train_set,test_set,hyper_param)
       
        # Test algorithm

        print(f'Ready to evaluate performances !')

        
        lossValues, labels = evaluation.foo(autoencoder, test_set, hyper_param)
        print(lossValues)
        #print(labels['labels'])

        AUCs.append(compute_metrics.get_AUC_score(labels, lossValues))
        print(f'AUC value stored !')
        print(AUCs)
    return AUCs
    


    