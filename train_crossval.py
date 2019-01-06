import os
import sys
import glob
import random
import numpy as np
np.random.seed(23087)
import pandas as pd
import tensorflow as tf
from keras import backend as k
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.models import Sequential, load_model
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout, Activation, Dense, Flatten
from keras.layers.convolutional import Convolution1D,AveragePooling1D,MaxPooling1D

'''
###################################
config = tf.ConfigProto()
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.5
# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))
###################################
'''
def train(argv):
    #Params
    epochs = 500
    batch_size = 512
    train_test_percent = 0.15 #optional
    folds = 10
    max_aug = int(argv[2])

    Mn_All,labels = load_data()

    for shift_aug_factor in range(1, max_aug+1):
        print("Performing {}x shifting data augmentation".format(shift_aug_factor))
        if argv[3] != None:
            root_path = os.path.join("weights","cross_validation_results", argv[3]+str("_Shift_dataaug-x")+str(shift_aug_factor))
            if not os.path.exists(root_path):
                os.mkdir(root_path)

        for fold in range(folds):
            model = build_neural_network_graph(graph_type=argv[1])
            (X_train, y_train), (X_test, y_test) = preprocess_crossval_aug(Mn_All, labels, shift_aug_factor, cv=True,
                                                fold=fold,n_splits=folds, crop_spectra=True, pca_aug = False)

            save_dir = os.path.join(root_path,"weights_"+str(fold))
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            best_model_file = save_dir+"/highest_val_acc_weights_epoch{epoch:02d}-val_acc{val_acc:.3f}_.h5"
            best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose = 1, save_best_only = True)
            hist = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                             nb_epoch=epochs, batch_size=batch_size,
                             callbacks = [best_model], shuffle = True, verbose=1)
            training_graphs(save_dir, hist)

        chemical_shift_test_acc=run_eval(root_path,Mn_All,labels,folds,shift_aug_factor)
        pd.DataFrame(chemical_shift_test_acc).to_csv(os.path.join(root_path,"chemical_shifts_acc.csv"), index=False, header=False)

def run_eval(root_path,Mn_All,labels,folds,shift_aug_factor):
    print( root_path)
    weight = load_best_weights(model=root_path)
    num_bins_translate = 100
    num_classes = 3

    total_confusion_matrix=np.zeros((num_classes,num_classes))
    acc = np.zeros(num_bins_translate)
    chemical_shift_test_acc= []
    for fold in range(folds):
        model = load_model(weight[fold])
        (X_train, y_train), (X_test, y_test) = preprocess_crossval_aug(Mn_All, labels, 0, cv=True, fold=fold,n_splits=folds, crop_spectra=False, pca_aug = False)
        run = chemical_shift_test(X_test, y_test, model, num_bins_translate)[:,1]

        acc += run
        print( run)
        total_confusion_matrix += confusion_matrix_generator(X_test[:,200:500], y_test, model)
        chemical_shift_test_acc.append(acc/len(weight))
    for i in range(len(chemical_shift_test_acc)):
        plot_chemical_shift_test(root_path, num_bins_translate, chemical_shift_test_acc[i])
    total_cv_confusion_matrix_generator(total_confusion_matrix)
    return chemical_shift_test_acc

def preprocess_crossval_aug(x, y, shift_aug_factor, cv=True, fold=None, n_splits=0, train_test_percent=0.25, crop_spectra=True, pca_aug = False):
    if cv == True:
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=n_splits, random_state=13, shuffle=False)
        X_train = [x[train_index] for train_index, test_index in cv.split(x, y)]
        X_test = [x[test_index] for train_index, test_index in cv.split(x, y)]
        y_train = [y[train_index] for train_index, test_index in cv.split(x, y)]
        y_test = [y[test_index] for train_index, test_index in cv.split(x, y)]
    else:
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=train_test_percent,
                                                        random_state=13, stratify=y)
    if fold != None:
        X_train, X_test, y_train, y_test = X_train[fold], X_test[fold], y_train[fold], y_test[fold]
        print("Samples will be from fold", (fold+1), " out of the", n_splits, " n_splits")
        print('Param train_test_percent will be ignored since folds are being used.')
    if pca_aug == True:
        from sklearn.decomposition import PCA

        noise = np.copy(X_train)
        mu = np.mean(noise, axis=0)
        pca = PCA()
        noise_model = pca.fit(noise)
        nComp = 10
        Xhat = np.dot(pca.transform(noise)[:,:nComp], pca.components_[:nComp,:])
        noise_level = np.dot(pca.transform(noise)[:,nComp:], pca.components_[nComp:,:])
        Xhat += mu

        snr_num = 200
        SNR = np.linspace(0,10,snr_num)
        noise_aug = []
        for i  in range(len(SNR)):
            noise_aug.append(SNR[i]*noise_level + Xhat)
            j = 0
            for spectra in noise_aug[i]:
                noise_aug[i][j] = spectra/np.max(spectra)
                j += 1
        X_train = np.array(noise_aug).reshape(snr_num*X_train.shape[0], X_train.shape[1])
        y_train = [item for i in range(snr_num) for item in y_train]

    lower_bound,upper_bound=200,500

    X_train, X_test, y_train, y_test = preprocess(X_train, X_test, y_train, y_test, lower_bound, upper_bound,
                                        shift_aug_factor=shift_aug_factor, crop=crop_spectra ,mean_center = True, norm = True )

    return (X_train, y_train), (X_test, y_test)

def preprocess(X_train, X_test, y_train, y_test, lower_bound, upper_bound, shift_aug_factor=None, crop=True, mean_center = False, norm = True):
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
    if shift_aug_factor != 0:
        print( "DATA AUG SHIFT")
        cropX_train,cropX_test=[],[]
        for i in range(len(X_train)):
            for j in range(shift_aug_factor):
                draw = int(random.random()*100)
                cropX_train.append(X_train[i,150+draw:450+draw])
        X_train = np.array(cropX_train).reshape(len(X_train)*shift_aug_factor, 300)
        X_test = X_test[:,lower_bound:upper_bound]
        y_train = sorted([item for i in range(shift_aug_factor) for item in y_train])
    elif crop==True and shift_aug_factor==1:
        X_train = X_train[:,lower_bound:upper_bound] #for test set, cropping to 635-665 eV, closer to qualitative
        X_test = X_test[:,lower_bound:upper_bound]   #for test set, cropping to 635-665 eV, closer to qualitative
    elif shift_aug_factor==0:
        pass
    else:
        X_train = X_train[:,lower_bound:upper_bound] #for test set, cropping to 635-665 eV, closer to qualitative
        X_test = X_test[:,lower_bound:upper_bound]   #for test set, cropping to 635-665 eV, closer to qualitative

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

def chemical_shift_test(X_test, y_test, model, num_bins_translate):
    accuracy_on_shifted_test_set = [model.evaluate(x = X_test[:,150+i:450+i], y = y_test, verbose=0 ) for i in range(num_bins_translate)]
    accuracy_on_shifted_test_set = np.array(accuracy_on_shifted_test_set)
    return accuracy_on_shifted_test_set

def plot_chemical_shift_test(path_to_output, num_bins_translate, acc1):
    delta_E = np.linspace(-5.,5.,100)
    plt.figure(1, figsize=(10,6))
    plt.plot(delta_E, acc1, 'b', label='Dense Network with Shift Aug.', linewidth = 2)
    plt.axvline(x=0, c = 'black')
    plt.ylabel('10-fold Cross Validation Test Accuracy')
    plt.xlabel('Chemical Shift (eV)')
    #plt.legend(loc = 'lower left')
    plt.savefig(os.path.join(path_to_output, 'chemical_shift_test.png'))
    plt.close()

def load_best_weights(model):
    root_path =  model
    weight_folds=sorted(next(os.walk(root_path))[1])
    weights=[]
    for fold in weight_folds:
        files_path = os.path.join(root_path, fold, '*.h5')
        cv_weights = sorted(glob.iglob(files_path), key=os.path.getctime, reverse=True)
        weights.append(cv_weights[0])
    return weights
def confusion_matrix_generator(X_test, y_test, model):
    y_test_pred, y_test_labels=[], []
    for i in range(len(X_test)):
        y_test_pred.append(np.argmax(model.predict(X_test[i:i+1])))
        y_test_labels.append(np.argmax(y_test[i]))
    print("Confusion Matrix of Test Set")
    conf_matrix = pd.DataFrame(confusion_matrix(y_pred=y_test_pred, y_true=y_test_labels))
    conf_matrix.columns = ["Mn2+", "Mn3+", "Mn4+" ]
    conf_matrix = pd.DataFrame.transpose(conf_matrix)
    conf_matrix.columns = ["Mn2+", "Mn3+", "Mn4+" ]
    conf_matrix = pd.DataFrame.transpose(conf_matrix)
    print(conf_matrix)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    return conf_matrix

def total_cv_confusion_matrix_generator(total_confusion_matrix):
    print("The averaged test-set accuracies for each class is: ")
    for i in range(total_confusion_matrix.shape[0]):
        print(total_confusion_matrix[i][i]/total_confusion_matrix[i].sum())
    print(total_confusion_matrix)
    print("Total test set accuracy:", np.diagonal(total_confusion_matrix).sum()/total_confusion_matrix.sum())
def load_data():
    #load data
    path_to_input = 'input_spectra'
    Mn2_C = pd.read_pickle(os.path.join(path_to_input, 'Mn2_615-685eV_thinnest_448.pkl'))
    Mn3_C = pd.read_pickle(os.path.join(path_to_input, 'Mn3_615-685eV_thin_765.pkl'))
    Mn4_C = pd.read_pickle(os.path.join(path_to_input, 'Mn4_615-685eV_thin_788.pkl'))
    Mn_All = (Mn2_C.append(Mn3_C, ignore_index=True)).append(Mn4_C, ignore_index=True)
    Mn_All = np.array(Mn_All)
    labels = make_labels(Mn2_C, Mn3_C, Mn4_C)
    return Mn_All, labels

def make_labels(Mn2_C, Mn3_C, Mn4_C):
    labels=[]
    for i in range(len(Mn2_C)):
        labels.append(0)
    for i in range(len(Mn3_C)):
        labels.append(1)
    for i in range(len(Mn4_C)):
        labels.append(2)
    return np.array(labels)

def build_neural_network_graph(graph_type):
    if graph_type == 'cnn':
        model = Sequential()
        activation = 'relu'
        model.add(Convolution1D(2, 9, input_shape=(300,1), activation=activation))
        model.add(BatchNormalization())
        model.add(AveragePooling1D())

        model.add(Convolution1D(2, 7, activation=activation))
        model.add(BatchNormalization())
        model.add(AveragePooling1D())

        model.add(Convolution1D(4, 7, activation=activation))
        model.add(BatchNormalization())
        model.add(AveragePooling1D())

        model.add(Convolution1D(8, 5, activation=activation))
        model.add(BatchNormalization())
        model.add(AveragePooling1D())

        model.add(Convolution1D(12, 3, activation=activation))
        model.add(BatchNormalization())
        model.add(AveragePooling1D())

        model.add(Dropout(0.85, seed=23087))
        model.add(Convolution1D(3, 1))
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

        model.add(Dense(3, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        print(model.summary())
        print("CNN Model created.")
        return model
    else:
        print('Custom Model')
        model = Sequential()
        activation = 'relu'
        model.add(Convolution1D(2, 9, input_shape=(300,1), activation=activation))
        model.add(BatchNormalization())
        model.add(AveragePooling1D())

        model.add(Convolution1D(2, 7, activation=activation))
        model.add(BatchNormalization())
        model.add(AveragePooling1D())

        model.add(Convolution1D(4, 7, activation=activation))
        model.add(BatchNormalization())
        model.add(AveragePooling1D())

        model.add(Convolution1D(8, 5, activation=activation))
        model.add(BatchNormalization())
        model.add(AveragePooling1D())

        model.add(Convolution1D(12, 3, activation=activation))
        model.add(BatchNormalization())
        model.add(AveragePooling1D())

        model.add(Flatten())

        model.add(Dropout(0.5, seed=23087, name='drop1'))
        model.add(Dense(8, activation='relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5, seed=23087, name='drop9'))
        model.add(Dense(4,activation='relu'))
        model.add(BatchNormalization())

        model.add(Dense(3, activation='softmax'))

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
    plt.close()
    plt.figure(figsize=(10, 8))

    plt.plot(hist.history['val_acc'], linewidth = 3)
    plt.plot(hist.history['acc'], linewidth = 3)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Test', 'Train'], loc='lower right')
    plt.savefig(os.path.join(save_dir, 'test_accuracy.png'))
    plt.close()

if __name__ == "__main__":
    train(sys.argv)
