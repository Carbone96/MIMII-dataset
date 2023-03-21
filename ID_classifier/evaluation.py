import numpy as np
import pandas as pd

def find_optimal_threshold(predictions, true_labels):
    thresholds = np.arange(0.8, 1, 0.0001)
    accs = []
    #print(f'An example of prediction by the model is : {predictions[4]}')
    for threshold in thresholds:

        #print(f'An example of predicted labels is : {predicted_labels}')
        predicted_labels = np.where(np.max(predictions, axis=1) < threshold, -1, np.argmax(predictions, axis=1))
        acc = accuracy_model(true_labels, predicted_labels)
        accs.append(acc)
    #print(f'The list of thresholds is : {accs}')
    optimal_threshold = thresholds[np.argmax(accs)]
    return optimal_threshold

def assign_label(test_data, test_labels, model, threshold = None):
    # Use the model to predict the probability of each class
    predictions = model.predict(test_data)
    #if threshold is None:
    #    threshold = find_optimal_threshold(predictions, test_labels)

    #predicted_labels = np.where(np.max(predictions) < threshold, -1, np.argmax(predictions, axis=1)) #assign -1  if below threshold

    return np.argmax(predictions, axis = 1)


def accuracy_model(y, y_pred):
    # Evaluate the model
    y = y[0].values.tolist()
    y_pred = [int(x) for x in y_pred]
    y = np.array(y)
    y_pred = np.array(y_pred)
    return np.sum(np.equal(y, y_pred)) / len(y)

def evaluate(test_data, test_labels, model):
    predicted_labels = assign_label(test_data, test_labels, model)
    return accuracy_model(test_labels, predicted_labels)
