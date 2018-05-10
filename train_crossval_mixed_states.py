import os
import sys
import numpy as np
np.random.seed(23087)
import pandas as pd
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.models import Sequential
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout, Activation, Dense, Flatten
from keras.layers.convolutional import Convolution1D,AveragePooling1D,MaxPooling1D
from sklearn.utils import class_weight

def train(argv):
    #Params
    epochs = 100
    batch_size = 2048
    train_test_percent = 0.15 #optional
    folds = 1

    if argv[2] != None:
        root_path = os.path.join("weights","cross_validation_results", argv[2])
        if not os.path.exists(root_path):
            os.mkdir(root_path)

    Mn_All,labels = load_data_mixed(num=1500)

    class_weights = class_weight.compute_class_weight('balanced', np.unique(labels), labels)
    class_weights = dict(enumerate(class_weights))

    for fold in range(folds):
        model = build_neural_network_graph(graph_type=argv[1])
        (X_train, y_train), (X_test, y_test) = preprocess_crossval_aug(Mn_All, labels, fold=fold, n_splits=folds, pca_aug = True)

        save_dir = os.path.join(root_path,"weights_"+str(fold))
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if folds == 1:
            (X_test,y_test)= load_ditized_spectra()
            best_model_file = save_dir+"/highest_val_acc_weights_epoch{epoch:02d}-valacc{val_acc:.3f}_.h5"
            best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose = 1, save_best_only = True)
            hist = model.fit(X_train, y_train,
                             nb_epoch=epochs, batch_size=batch_size,
                             callbacks = [best_model], validation_data=(X_test, y_test),
                             class_weight = class_weights, shuffle = True, verbose=1)
        else:
            best_model_file = save_dir+"/highest_val_acc_weights_epoch{epoch:02d}-valacc{val_acc:.3f}_.h5"
            best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose = 1, save_best_only = True)
            hist = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                             nb_epoch=epochs, batch_size=batch_size,
                             callbacks = [best_model],
                             class_weight = class_weights, shuffle = True, verbose=1)
        training_graphs(save_dir, hist)


def load_data_mixed(num=1000):
    path_to_input = 'input_spectra'
    Mn2_C = np.array(pd.read_pickle(os.path.join(path_to_input, 'Mn2_615-685eV_thinnest_448.pkl')))
    Mn3_C = np.array(pd.read_pickle(os.path.join(path_to_input, 'Mn3_615-685eV_thin_765.pkl')))
    Mn4_C = np.array(pd.read_pickle(os.path.join(path_to_input, 'Mn4_615-685eV_thin_788.pkl')))
    Mn23_2 = apply_mixed_aug(Mn2_C,Mn3_C, 0.5, 0.5, num)

    Mn34_2 = apply_mixed_aug(Mn3_C,Mn4_C, 0.5, 0.5, num)

    Mn_All=np.concatenate((Mn2_C[:,200:500],
                    Mn23_2,
                    Mn3_C[:,200:500],
                    Mn34_2,
                    Mn4_C[:,200:500]))

    labels = ([0]*len(Mn2_C)  +
         [1]*len(Mn23_2) +
         [2]*len(Mn3_C)  +
         [3]*len(Mn34_2) +
         [4]*len(Mn4_C)  )
    labels = np.array(labels)
    return Mn_All, labels
'''
def load_data_mixed(num):
    path_to_input = 'input_spectra'
    Mn2_C = np.array(pd.read_pickle(os.path.join(path_to_input, 'Mn2_615-685eV_thinnest_448.pkl')))
    Mn3_C = np.array(pd.read_pickle(os.path.join(path_to_input, 'Mn3_615-685eV_thin_765.pkl')))
    Mn4_C = np.array(pd.read_pickle(os.path.join(path_to_input, 'Mn4_615-685eV_thin_788.pkl')))

    Mn23_1 = apply_mixed_aug(Mn2_C,Mn3_C, 0.33, 0.66, num)
    Mn23_2 = apply_mixed_aug(Mn2_C,Mn3_C, 0.66, 0.33, num)

    Mn34_1 = apply_mixed_aug(Mn3_C,Mn4_C, 0.33, 0.66, num)
    Mn34_2 = apply_mixed_aug(Mn3_C,Mn4_C, 0.66, 0.33, num)

    Mn_All=np.concatenate((Mn2_C[:,200:500],
                    Mn23_1,
                    Mn23_2,
                    Mn3_C[:,200:500],
                    Mn34_1,
                    Mn34_2,
                    Mn4_C[:,200:500]))

    labels = ([0]*len(Mn2_C)  +
         [1]*len(Mn23_1) +
         [2]*len(Mn23_2) +
         [3]*len(Mn3_C)  +
         [4]*len(Mn34_1) +
         [5]*len(Mn34_2) +
         [6]*len(Mn4_C)  )
    labels = np.array(labels)
    return Mn_All, labels
'''
def apply_mixed_aug(Mn_1, Mn_2, Mn_1_frac, Mn_2_frac, num):
    Mn_sum_list = []
    for i in range(num):
        rn1 = np.random.choice(len(Mn_1))
        rn2 = np.random.choice(len(Mn_2))
        rn_crop1 = np.random.choice(np.arange(-10, 10))
        rn_crop2 = np.random.choice(np.arange(-10, 10))

        spectra1 = Mn_1[rn1][200+rn_crop1: 500+rn_crop1]
        spectra2 = Mn_2[rn2][200+rn_crop2: 500+rn_crop2]

        Mn_sum = Mn_1_frac*spectra1 + Mn_2_frac*spectra2
        Mn_sum /= np.max(Mn_sum)
        Mn_sum_list.append(Mn_sum)
    Mn_sum_list = np.array(Mn_sum_list)
    return Mn_sum_list

def preprocess_crossval_aug(x, y, fold=None, n_splits=0, train_test_percent=0.25, pca_aug = False):
    if n_splits > 3:
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=n_splits, random_state=13, shuffle=False)
        X_train = [x[train_index] for train_index, test_index in cv.split(x, y)]
        X_test = [x[test_index] for train_index, test_index in cv.split(x, y)]
        y_train = [y[train_index] for train_index, test_index in cv.split(x, y)]
        y_test = [y[test_index] for train_index, test_index in cv.split(x, y)]

        X_train, X_test, y_train, y_test = X_train[fold], X_test[fold], y_train[fold], y_test[fold]
        print("Samples will be from fold", (fold+1), " out of the", n_splits, " n_splits")
        print('Param train_test_percent will be ignored since folds are being used.')
    elif n_splits == 1:
        X_train = x
        y_train = y
        X_test = np.zeros((x.shape[0],x.shape[1]))
        y_test = y
    elif n_splits == 2:
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=train_test_percent,
                                                        random_state=13, stratify=y)
    if pca_aug == True:
        X_train, y_train = apply_pca_aug(X_train, y_train, snr_steps=25)

    X_train, X_test, y_train, y_test = preprocess(X_train, X_test, y_train, y_test, mean_center = True, norm = True )
    return (X_train, y_train), (X_test, y_test)

def apply_pca_aug(X_train, y_train, snr_steps):
    from sklearn.decomposition import PCA
    noise = np.copy(X_train)
    mu = np.mean(noise, axis=0)
    pca = PCA()
    noise_model = pca.fit(noise)
    nComp = 10
    Xhat = np.dot(pca.transform(noise)[:,:nComp], pca.components_[:nComp,:])
    noise_level = np.dot(pca.transform(noise)[:,nComp:], pca.components_[nComp:,:])
    Xhat += mu

    SNR = np.linspace(0,5,snr_steps)
    noise_aug = []
    for i  in range(len(SNR)):
        noise_aug.append(SNR[i]*noise_level + Xhat)
        j = 0
        for spectra in noise_aug[i]:
            noise_aug[i][j] = spectra/np.max(spectra)
            j += 1
    X_train = np.array(noise_aug).reshape(snr_steps*X_train.shape[0], X_train.shape[1])
    y_train = [item for i in range(snr_steps) for item in y_train]
    return X_train, y_train

def load_ditized_spectra():
    data_path = '/home/mike/Mn_Valences/Mn_Classifier_CV_Good_Copy/Data/Digitized_Mn_Usecases.pkl'
    data = pd.read_pickle(data_path)
    benchmark = np.concatenate((data[1], data[2], data[0]))
    labels = [0]*len(data[1])+ [2]*len(data[2])+ [4]*len(data[0])

    X_test = np.zeros( (len(benchmark),300) )
    i=0
    for spectra in benchmark:
        x = spectra['Energy']
        y = spectra['Intensity']
        min_energy, max_energy= np.min(x), np.max(x)
        new_energy=np.linspace(min_energy,max_energy,300)

        new_intensity = np.interp(new_energy, x, y)
        new_intensity -= np.mean(new_intensity)
        new_intensity /= np.max(new_intensity)
        X_test[i] = new_intensity
        i+=1
    X_test = X_test.reshape(X_test.shape + (1,))

    y_test = np.array(labels)
    y_test = np_utils.to_categorical(y_test)
    return (X_test,y_test)

def preprocess(X_train, X_test, y_train, y_test, mean_center = False, norm = True):
    X_train = np.array(X_train).astype('float32')
    X_test = np.array(X_test).astype('float32')

    if mean_center == True:
        X_train -=  np.mean(X_train)
        X_test -= np.mean(X_test)
        print( 'Data mean-centered')
    if norm == True:
            X_train /= np.max(X_train)
            X_test /= np.max(X_test)
            print( 'Data normalized')

    X_test = X_test.reshape(X_test.shape + (1,))
    X_train = X_train.reshape(X_train.shape + (1,))

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    print( 'Data one-hot encoded')

    print("Total of "+str(y_test.shape[1])+" classes.")
    print("Total of "+str(len(X_train))+" training samples.")
    print("Total of "+str(len(X_test))+" testing samples.")

    return X_train, X_test, y_train, y_test


def build_neural_network_graph(graph_type):
    if graph_type == 'cnn':
        model = Sequential()
        activation = 'relu'

        model.add(Convolution1D(2, 9, input_shape=(300,1)))
        model.add(BatchNormalization())
        model.add(Activation(activation))
        model.add(AveragePooling1D())

        model.add(Convolution1D(2, 7))
        model.add(BatchNormalization())
        model.add(Activation(activation))
        model.add(AveragePooling1D())

        model.add(Convolution1D(4, 7))
        model.add(BatchNormalization())
        model.add(Activation(activation))
        model.add(AveragePooling1D())

        model.add(Convolution1D(8, 5))
        model.add(BatchNormalization())
        model.add(Activation(activation))
        model.add(AveragePooling1D())

        model.add(Convolution1D(12, 3))
        model.add(BatchNormalization())
        model.add(Activation(activation))
        model.add(AveragePooling1D())

        model.add(Dropout(0.1, seed=23087))
        model.add(Convolution1D(5, 1))
        model.add(BatchNormalization())
        model.add(GlobalAveragePooling1D())

        model.add(Activation('softmax', name='loss'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        print(model.summary())
        print("CNN Model created.")
        return model

    elif graph_type=='MLP':
        model = Sequential()

        model.add(Flatten(input_shape=(300,1)))

        model.add(Dropout(0.5, seed=23087, name='drop1'))
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5, seed=23087, name='drop9'))
        model.add(Dense(32,activation='relu'))
        model.add(BatchNormalization())

        model.add(Dense(5, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        print(model.summary())
        print("CNN Model created.")
        return model
    else:
        print('Custom Model')
        model = Sequential()
        activation = 'relu'
        model.add(Convolution1D(4, 9, input_shape=(300,1), activation=activation))
        model.add(BatchNormalization())
        model.add(AveragePooling1D())

        model.add(Convolution1D(4, 7, activation=activation))
        model.add(BatchNormalization())
        model.add(AveragePooling1D())

        model.add(Convolution1D(8, 7, activation=activation))
        model.add(BatchNormalization())
        model.add(AveragePooling1D())

        model.add(Convolution1D(16, 5, activation=activation))
        model.add(BatchNormalization())
        model.add(AveragePooling1D())

        model.add(Convolution1D(32, 3, activation=activation))
        model.add(BatchNormalization())
        model.add(AveragePooling1D())

        model.add(Flatten())

        model.add(Dropout(0.5, seed=23087, name='drop1'))
        model.add(Dense(16, activation='relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5, seed=23087, name='drop9'))
        model.add(Dense(16,activation='relu'))
        model.add(BatchNormalization())

        model.add(Dense(5, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        print(model.summary())
        print("CNN Model created.")
        return model

def training_graphs(save_dir, hist):
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
    plt.savefig(os.path.join(save_dir, 'training_accuracy.png'))

    plt.figure(figsize=(10, 8))

    plt.plot(hist.history['val_acc'], linewidth = 3)
    plt.plot(hist.history['acc'], linewidth = 3)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Test', 'Train'], loc='lower right')
    plt.savefig(os.path.join(save_dir, 'test_accuracy.png'))


if __name__ == "__main__":
    train(sys.argv)
