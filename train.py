
import os
import sys
import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.models import Sequential
from matplotlib import pyplot as plt
from keras.callbacks import ModelCheckpoint
from sklearn.cross_validation import train_test_split
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout, Activation, Dense, Flatten
from keras.layers.convolutional import Convolution1D,AveragePooling1D,MaxPooling1D

def train(argv):
    #Params
    epochs = 1000
    batch_size = 512
    seed = 7
    train_test_percent = 0.15

    #load dataset and preprocess it, formatting it to a readable tensor
    X_train, X_test, y_train, y_test = load_data_preprocess(train_test_percent, crop_spectra=True, aug = False)

    if argv[1] == 'cnn':
        model = build_cnn_architecture()
    if argv[1] == 'mlp':
        model = build_mlp_architecture()

    # fit and run our model
    np.random.seed(seed)
    best_model_file = "weights/CNN_Noise_DataAug/highest_val_acc_weights_epoch{epoch:02d}-val_acc{val_acc:.3f}_"+str(argv[1])+".h5"
    best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose = 1, save_best_only = True)
    hist = model.fit(X_train,
                     y_train,
                     validation_data=(X_test, y_test),
                     nb_epoch=epochs,
                     batch_size=batch_size,
                     callbacks = [best_model],
                     shuffle = True,
                     verbose=1)

    print("done training")
    training_graphs(hist)

def load_data_preprocess(train_test_percent, crop_spectra=True, aug = False):
    #load data
    path_to_input = 'input_spectra'
    Mn2_C = pd.read_pickle(os.path.join(path_to_input, 'Mn2_Larger_Clean_Thin.pkl'))
    Mn3_C = pd.read_pickle(os.path.join(path_to_input, 'Mn3_Larger_Clean_Thin.pkl'))
    Mn4_C = pd.read_pickle(os.path.join(path_to_input, 'Mn4_Larger_Clean_Thin.pkl'))
    Mn_All = (Mn2_C.append(Mn3_C, ignore_index=True)).append(Mn4_C, ignore_index=True)
    Mn_All = np.array(Mn_All)

    labels = make_labels(Mn2_C, Mn3_C, Mn4_C)

    X_train, X_test, y_train, y_test = train_test_split(Mn_All, labels, test_size=train_test_percent, random_state=13)
    if aug == True:
        from sklearn.decomposition import PCA
        noise_aug = []
        noise = np.copy(X_train)
        mu = np.mean(noise, axis=0)
        pca = PCA()
        noise_model = pca.fit(noise)
        nComp = 10
        Xhat = np.dot(pca.transform(noise)[:,:nComp], pca.components_[:nComp,:])
        noise_level = np.dot(pca.transform(noise)[:,nComp:], pca.components_[nComp:,:])
        Xhat += mu
        SNR = np.linspace(1,5,50)
        for i  in range(len(SNR)):
            noise_aug.append(SNR[i]*noise_level + Xhat)
            j = 0
            for spectra in noise_aug[i]:
                noise_aug[i][j] = spectra/np.max(spectra)
                j += 1
        X_train = np.array(noise_aug).reshape(50*2684,700)
        y_train = [item for i in range(50) for item in y_train]

    if crop_spectra == True:
        X_train, X_test = crop(X_train, X_test, min = 100, max = 600)

    X_train, X_test, y_train, y_test = preprocess(X_train, X_test, y_train, y_test, mean_center = True, norm = True )
    return X_train, X_test, y_train, y_test

def make_labels(Mn2_C, Mn3_C, Mn4_C):
    labels=[]
    for i in range(len(Mn2_C)):
        labels.append(0)
    for i in range(len(Mn3_C)):
        labels.append(1)
    for i in range(len(Mn4_C)):
        labels.append(2)
    return labels

def crop(X_train, X_test, min = 100,max = 600):
    crop_X_train = X_train[:,min:max]
    crop_X_test = X_test[:,min:max]
    return crop_X_train, crop_X_test

def preprocess(X_train, X_test, y_train, y_test, mean_center = False, norm = True):
    X_train = np.array(X_train).astype('float32')
    X_train = X_train.reshape(X_train.shape + (1,))
    X_test = np.array(X_test).astype('float32')
    X_test = X_test.reshape(X_test.shape + (1,))
    if mean_center == True:
        X_train -=  np.mean(X_train)
        X_test -= np.mean(X_test)
        print( 'Data mean-centered')
    if norm == True:
        X_train /= np.max(X_train)
        X_test /= np.max(X_test)
        print( 'Data normalized')

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    print( 'Data one-hot encoded')

    print("Total of "+str(y_test.shape[1])+" classes.")
    print("Total of "+str(len(X_train))+" training samples.")
    print("Total of "+str(len(X_test))+" testing samples.")

    return X_train, X_test, y_train, y_test


def build_cnn_architecture():
    model = Sequential()
    activation = 'relu'
    model.add(Convolution1D(2, 9, input_shape=(500,1), activation=activation))
    model.add(AveragePooling1D())
    model.add(BatchNormalization())

    model.add(Convolution1D(2, 7, activation=activation))
    model.add(AveragePooling1D())
    model.add(BatchNormalization())

    model.add(Convolution1D(4, 7, activation=activation))
    model.add(AveragePooling1D())
    model.add(BatchNormalization())

    model.add(Convolution1D(4, 5, activation=activation))
    model.add(AveragePooling1D())
    model.add(BatchNormalization())

    model.add(Convolution1D(8, 3, activation=activation))
    model.add(AveragePooling1D())
    model.add(BatchNormalization())

    model.add(Dropout(0.10))
    model.add(Convolution1D(3, 1))
    model.add(GlobalAveragePooling1D())

    model.add(Activation('softmax', name='loss'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())
    print("CNN Model created.")
    return model

def build_mlp_architecture():
    model = Sequential()

    model.add(Flatten(input_shape=(500,1)))

    model.add(Dropout(0.5, name='drop1'))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dropout(0.5, name='drop9'))
    model.add(Dense(64,activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())
    print("CNN Model created.")
    return model

def training_graphs(hist):
    #summarize history for accuracy
    plt.figure(figsize=(15, 5))
    plt.rcParams.update({'font.size': 16})

    plt.subplot(1, 2, 1)
    plt.plot(hist.history['acc'], linewidth = 3)
    plt.title('Model Training Accuracy')
    plt.ylabel('Training Accuracy')
    plt.xlabel('Epoch')

    # summarize history for loss
    plt.subplot(1, 2, 2)
    plt.plot(hist.history['loss'], linewidth = 3)
    plt.title('Model Training Loss')
    plt.ylabel('Cross Entropy Loss')
    plt.xlabel('Epoch')
    plt.savefig('figures/training_accuracy.png')
    plt.show()

    plt.figure(figsize=(10, 8))

    plt.plot(hist.history['val_acc'], linewidth = 3)
    plt.plot(hist.history['acc'], linewidth = 3)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Test', 'Train'], loc='lower right')
    plt.savefig('figures/test_accuracy.png')

    plt.show()


if __name__ == "__main__":
    train(sys.argv)
