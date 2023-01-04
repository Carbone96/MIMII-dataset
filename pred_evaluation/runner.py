import compute_metrics
import evaluation
import learner
import import_processed_data


def run(data_folder : str, run_settings :dict, data_settings :dict):

    print(f'starting the import !')
    train_set, test_set, labels, run_settings['raws_per_file'] = import_processed_data.import_data(data_folder, data_settings)

    print(test_set)

    autoencoder = learner.createLearner(train_set,test_set,labels, run_settings)
    
    # Test algorithm
    print(f'Ready to evaluate performances !')
    anomaly_scores = evaluation.evaluate(autoencoder,train_set, test_set,labels, run_settings)


    print(f'starting the evaluation!')
    """
    warning : the labels should be inversed in the datasets, to compensate I take 1-AUC for now.
    """
    auc = compute_metrics.calculate_AUC(labels, anomaly_scores)
    print(auc)
    print(f'AUC value stored !')

    return auc
    


    