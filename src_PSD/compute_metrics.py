from sklearn import metrics

def get_AUC_score(method_name : str, test_labels, lossValues):
    if method_name == 'spectro':
        
        return metrics.roc_auc_score(test_labels, lossValues)

    if method_name == 'psd':
        return metrics.roc_auc_score(test_labels, lossValues)
        
    else: 
        pass
