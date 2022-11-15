import matplotlib.pyplot as plt
import random
import librosa
import librosa.display
import tensorflow as tf
import keras
import sklearn
from sklearn import metrics
import numpy as np

def epoch_losses(history):
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Reconstruction error")
    plt.legend()
    

def reconstruction(df,autoencoder):
    
    NumberOfRaws = df.shape[0]
    NumberOfColumns = df.shape[1]
    Xvector = np.arange(NumberOfColumns)
    rawID = random.randint(0,NumberOfRaws-1)
    selectedData = df[rawID]
    
    encoded_data = autoencoder.encoder(selectedData).numpy()
    reconstructedData = autoencoder.decoder(encoded_data).numpy()
    
    plt.clf()
    plt.plot(selectedData, 'b')
    plt.plot(reconstructedData, 'g')
    plt.fill_between(Xvector, reconstructedData, selectedData, color='lightgreen')
    plt.legend(labels=["Input", "Reconstruction", "Error"])
    plt.title("Reconstruction error : standard data")
    plt.show()
    

def SpectroReconstruct(df,autoencoder,spectroShape = (309,320)):
    
    # Get data in list-format
    originalData = df[random.randint(0,df.shape[0])].to_numpy()
    reconstructions = autoencoder.predict(originalData)
    
    # Reshape in spectroShape-format for plots
    originalData = originaldata.reshape(spectroShape(0),spectroShape(1))
    reconstructedData = reconstructions.reshape(spectroShape(0),spectroShape(1))
    
    # Reconstructed data
    fig, ax = plt.subplots()
    img = librosa.display.specshow(reconstructedData, x_axis='time',y_axis='mel', sr=16000, ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title='Mel-freq spectro : reconstructed data')
    # Original data
    fig, ax = plt.subplots()
    img = librosa.display.specshow(originalData, x_axis='time',y_axis='mel', sr=16000, ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title='Mel-freq spectro : original data')
    
def reconstructionErrorHist(df_anomalous,train_set,autoencoder):
    # Compute reconstructions error on HEALTHY df
    reconstructions = autoencoder.predict(train_set)
    healthy_loss = tf.keras.losses.mae(reconstructions,train_set)
    
    # Compute reconstructions error on ABNORMAL df
    reconstructions_anomalous = autoencoder.predict(df_anomalous)
    anomalous_loss = tf.keras.losses.mae(reconstructions_anomalous, anomalous_test_data)


    plt.hist(healthy_loss[None,:], bins=50)
    plt.hist(anomalous_loss[None,:], bins=50)
    plt.xlabel("Train loss")
    plt.ylabel("No of examples")
    plt.show()

    
### Function used for AUC computation :
def lossCalcMSE(model,data):
    reconstructions = model(data)
    return tf.keras.losses.mse(reconstructions,data)

def lossCalcMAE(model,data):
    reconstructions = model(data)
    return tf.keras.losses.mae(reconstructions,data)



def AUC(test_labels,test_set,model,metric = "mse"):
    
    if metric == "mse":
        lossValues = lossCalcMSE(model,test_set)
    if metric == "mae":
        lossValues = lossCalcMAE(mode,test_set)
        
    auc = 1 - metrics.roc_auc_score(test_labels, lossValues)
    false_positive_rate, true_positive_rate, thresolds = metrics.roc_curve(~test_labels, lossValues)
    
    
    plt.figure(figsize=(10, 8), dpi=100)
    plt.axis('scaled')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title("AUC & ROC Curve")
    plt.plot(false_positive_rate, true_positive_rate, 'g')
    plt.fill_between(false_positive_rate, true_positive_rate, facecolor='lightgreen', alpha=0.7)
    plt.text(0.95, 0.05, 'AUC = %0.4f' % auc, ha='right', fontsize=12, weight='bold', color='blue')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()
    
    