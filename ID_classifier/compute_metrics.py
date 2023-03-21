from sklearn import metrics

def calculate_AUC(test_labels, loss_values):
    """
    Calculate and return the Area Under the Curve (AUC) score for the given test labels and loss values.
    """
    return metrics.roc_auc_score(test_labels, loss_values)

def calculate_accuracy(test_labels, loss_values):
    """
    Calculate and return the accuracy score for the given test labels and loss values.
    """
    return metrics.accuracy_score(test_labels, loss_values)

