from src_PSD import processing_methods
from src_PSD import preprocessing
from src_PSD import learner
from src_PSD import compute_metrics
from src_PSD import splitter
from src_PSD import evaluation

CHANNELS = {'fan' : 5,'pump': 3,'valve' : 1, 'slider' : 7, 'test' : 5} # il faudrait vérifier que je sélectionne les bons channels, pour le moment ça fera l'affaire
IDS = [['id_00'],['id_02'], ['id_04'], ['id_06']]
EPOCHS = 50
BATCH = 512

def foo(IDchosen : int, machine_name : str, user_path : str, base_folder : str, method_name : str):
    
    
    IDs_chosen = IDS[IDchosen] 
    channel = CHANNELS[machine_name]
    AUCs = []

    for IDindex, IDval in enumerate(IDs_chosen):
        
        datapath = generate_filepath(base_folder,IDval, 'normal', machine_name)
        print(datapath)
        df_normal,dic_file_normal = processing_methods.foo(user_path, datapath, channel, method_name )
        
        datapath = generate_filepath(base_folder,IDval, 'abnormal', machine_name)
        df_abnormal,dic_file_abnormal = processing_methods.foo(user_path, datapath, channel, method_name)
        
        print(f'Data import has been successful !')
        
        # Create datasets for training/evaluation
        train_set, test_set, test_labels = splitter.foo(df_normal,df_abnormal)
        print(f' test_labels_values {test_labels.sort_index(axis=0)[0:15]}')
        # Preprocessing 
        train_set, test_set, min_val, max_val = preprocessing.MinMax_foo(train_set, test_set)
    
    
        print(f'Data preprocessing is finished.. beginning learning !')
        # Train algorithm
        autoencoder = learner.MIMII_AE(train_set.shape[1])
        history = autoencoder.fit(train_set, train_set, 
                                  epochs=EPOCHS, 
                                  batch_size=BATCH,
                                  validation_data=(test_set,test_set),
                                  validation_split = 0.1,
                                  verbose = 0,
                                  shuffle=False
                                 )

        lossValues = evaluation.foo(method_name, autoencoder, test_set, dic_file_abnormal)
        AUCs.append(compute_metrics.get_AUC_score(method_name, test_labels, lossValues))
        print(f'AUC value stored !')
    return AUCs
    
def generate_filepath(base_folder : str, ID : int,status : str, machine_name : str) -> str:
    """ Generate the filepath to get the audio file """
    
    return base_folder + machine_name + '/' + ID + '/' + status

    