# Import necessary libraries
from tensorflow import keras
import numpy as np

def classifier_training(train_data, train_labels, num_IDs = 4, epochs = 1):

    # One-hot encode the labels
    train_labels = keras.utils.to_categorical(train_labels, num_IDs)
    # Build the model
    model = keras.Sequential()
    # Add convolutional layers
    model.add(keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(313, 64, 1)))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(keras.layers.GlobalAveragePooling2D())
    # Add a softmax output layer with the number of classes
    model.add(keras.layers.Dense(num_IDs, activation='softmax'))
    # Compile the model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    # Train the model
    model.fit(train_data, train_labels, epochs=epochs)
    return model

def classifier(train_data, train_labels):
    # Run the training and validation
    model = classifier_training(train_data, train_labels)
    return model