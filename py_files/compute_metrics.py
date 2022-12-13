from sklearn import metrics
import pandas as pd

def get_AUC_score(test_labels, lossValues):
        
    return metrics.roc_auc_score(test_labels,lossValues)
