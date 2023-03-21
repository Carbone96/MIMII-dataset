import numpy as np

def get_model_softmax_probabilities(model, test_data):
    # For each Data Vector, return the softmax probabilities
    return model.predict(test_data)

def assign_label_to_softmax_probabilities(softmax_probabilities):
    # Assign a label (= the argmax) to each softmax probality vector
    return np.argmax(softmax_probabilities, axis = 1)

def predict(model ,test_data):
    softmax_proba = get_model_softmax_probabilities(model, test_data)
    return assign_label_to_softmax_probabilities(softmax_proba)