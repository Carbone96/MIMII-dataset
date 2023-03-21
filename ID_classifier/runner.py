
import import_processed_data
import classifier
import formatting
import evaluation

def run(data_folder : str, run_settings :dict, data_settings :dict):

    train_set, test_set, train_labels, test_labels, hyper_param = import_processed_data.import_data(data_folder, data_settings)
    
    print([test_labels == 0])
    #print(f'The test labels are : {test_labels[test_labels == 0]}')


    """
    print('Training set')
    print(train_set.head())

    print('test set')
    print(test_set.head())

    print('train label set')
    print(train_labels.head())
    print(type(train_labels.iloc[500].values[0]))

    print('test label set')
    print(f'the 500 value of test_label is {test_labels.iloc[500].values[0]} its type is {type(test_labels.iloc[500].values[0])}')
    """
    
    np_train_set , np_test_set = formatting.CNNformat(train_set, test_set, hyper_param['raws_per_file'])

    

    model = classifier.classifier_training(np_train_set, train_labels)

    
    

    accuracy = evaluation.evaluate(np_test_set, test_labels, model)
    print(f"testing accuracy is : {accuracy}")


    