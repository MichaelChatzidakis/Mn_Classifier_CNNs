import os
import sys
import numpy as np
import pandas as pd
from keras import backend as K
from keras.utils import np_utils
from keras.models import load_model
from sklearn.metrics import log_loss
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from train import load_data_preprocess
from sklearn.metrics import confusion_matrix
import glob

weights_folder = 'weights/CNN_Noise_DataAug'
files_path = os.path.join(weights_folder, '*')
weights = sorted(
                glob.iglob(files_path),
                key=os.path.getctime,
                reverse=True)

#Load Data
path_to_output = 'figures'
X_train, X_test, y_train, y_test = load_data_preprocess(train_test_percent=0.15, crop_spectra=False)
#we need the uncropped spectra for the chemical shift test
def test_model(argv):
    #Load Model
    if argv[1] == 'cnn':
        print( 'CNN model loading...')
        model = load_model('weights/highest_val_acc_weights_cnn.h5')
        model.summary()
        plot_chemical_shift_test(model)
    elif argv[1] =='both':
        print( 'MLP and CNN models loading...')
        model = load_model('weights/highest_val_acc_weights_cnn.h5')
        model_MLP = load_model('weights/highest_val_acc_weights_mlp.h5')
        plot_chemical_shift_test(model, model_MLP)

    elif argv[1] =='data_aug_cnn':
        print( 'CNN with and without data aug models loading...')
        model = load_model('weights/highest_val_acc_weights_cnn.h5')
        model_NoAug = load_model('weights/highest_val_acc_weights_cnn_noAug.h5')
        plot_chemical_shift_test(model, model_NoAug)
        plot_noise_test(model, spectra= 451)
        plot_noise_test(model_NoAug, spectra= 451)
        plt.show()
    elif argv[1].startswith('last') == True:
        if len(argv[1].split('-')) == 1:
            print(weights[0], "---weights used")
            model = load_model(weights[0])
        else:
            snapshot = int(argv[1].split('-')[1])
            print(weights[snapshot], "---weights used")
            model = load_model(weights[snapshot])

        model.summary()
        plot_chemical_shift_test(model)
    else:
        print('Model not found.', argv[1])

    X_train, X_test, y_train, y_test = load_data_preprocess(train_test_percent=0.15, crop_spectra=True)

    plot_noise_test(model, spectra= 451)
    confusion_matrix_generator(X_test, y_test, model)
    layer_output_Mn2, layer_output_Mn3, layer_output_Mn4 = [], [], []
    get_activations(model, layer_output_Mn2, X_train[1200:1201], norm=True)
    get_activations(model, layer_output_Mn3, X_train[300:301], norm=True)
    get_activations(model, layer_output_Mn4, X_train[500:501], norm=True)


    layer_num = [0,3,6,9,12,16]
    filter_num = [2,2,2,4,8,3]
    i=0
    for layer in layer_num:
        print(i)
        print(layer)
        plot_activations_at_layers(model, layer_output_Mn4, layer, filter_num[0], X_train[500])
        #plt.plot(np.linspace(620,660,500), X_train[500])
        i+=1
    plt.show()



def get_activations(model, layer_output, X_train, norm=False):
    layers = [n for n in range(len(model.layers))]
    for layer in layers:
        get_layer_output = K.function([model.layers[0].input, K.learning_phase()],[model.layers[layer].output])
        layer_output.append(get_layer_output([X_train, 0])[0][0] )
    if norm == True:
        for i in range(len(layer_output)-2):
            for j in range(layer_output[i].shape[1]):
                layer_output[i][:,j] -= np.min(layer_output[i][:,j])
                layer_output[i][:,j] /= np.max(layer_output[i][:,j])


def plot_activations_at_layers(model, layer_output, layer_num, num_filters, X_train):
    plt.figure(figsize=(4, 12))
    label = ['Mn2+', 'Mn3+', 'Mn4+']
    valence = label[np.argmax(layer_output[-1])]
    name = str(model.layers[layer_num]).split()[0].split('.')[3] + ' Activations- Label: '+str(valence)
    save_name = str(name)+'_'+str(valence)+'.png'
    if layer_num != 16:
        label = ['Filter 1', 'Filter 2', 'Filter 3']
        valence = label[np.argmax(layer_output[-1])]
        name = 'Convolution Activations (Layer '+str(layer_num)+')'
        save_name = name + '.png'

    num_filters = layer_output[layer_num].shape[1]
    offset = []
    for i in range(num_filters):
        offset.append(max(layer_output[layer_num][:,i]))
        shift = 2*(sum(offset) - min(layer_output[layer_num][:,i]))

        plt.plot(np.linspace(620, 660, len(layer_output[layer_num][:,i])), layer_output[layer_num][:,i]+shift, 'black', label=str(i), marker='.')
        if layer_num != 16:
            plt.axhline(shift, c = 'black')
    #plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.title(name)
    #plt.ylabel('Intensity')
    #plt.xlabel('Energy (eV)')
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_output, save_name))

def confusion_matrix_generator(X_test, y_test, model):
    y_test_pred, y_test_labels=[], []
    for i in range(0, len(X_test)):
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


def plot_noise_test(model, spectra):
    SNR,noise_acc_CNN, noise_test = noise_PCA_spectra(model)

    fig = plt.figure(4, figsize=(10,6))
    ax = fig.add_subplot(111)
    plt.rcParams.update({'font.size': 16})

    for i in range(5):
        plt.plot(np.linspace(620,660,500), noise_test[i*10][spectra,]+1.5*i, linewidth=1.5)
    plt.yticks([])
    plt.ylim(-0.4,7.2)
    ax.annotate('Original',
                xy=(622, 0), xycoords='data')
    ax.annotate('2x Noise',
                xy=(622, 1.6), xycoords='data')
    ax.annotate('3x Noise',
                xy=(622, 3.2), xycoords='data')
    ax.annotate('4x Noise',
                xy=(622, 4.7), xycoords='data')
    ax.annotate('5x Noise',
                xy=(622, 6.3), xycoords='data')
    plt.xlabel('Energy (eV)')
    plt.savefig(os.path.join(path_to_output, 'input_spectra_with_varying_noise.png'))
    plt.figure(3, figsize=(10,6))
    plt.plot(SNR, noise_acc_CNN[:,1], '.', markersize=15)
    plt.ylabel('Accuracy')
    plt.xlim(0.95, 5.05)
    plt.xticks([1,2,3,4,5])
    plt.yticks([0.60, 0.70, 0.80, 0.90, 1.0])
    plt.xlabel('Multiples of Noise Added')
    plt.savefig(os.path.join(path_to_output, 'accuracy_vs_noise.png'))

def noise_PCA_spectra(model):
    noise_test, noise_acc = [], []
    noise = np.copy(X_test[:,100:600,0])
    mu = np.mean(noise, axis=0)

    pca = PCA()
    noise_model = pca.fit(noise)
    '''
    plt.figure(5)
    plt.title('Scree Plot')
    plt.plot(pca.explained_variance_ratio_, 'r.')
    plt.xlim(-0.1,30)
    plt.ylim(-0.01, 0.6)
    plt.xlabel('Principal Components')
    plt.savefig(os.path.join(path_to_output, 'Scree-Plot-for-Noise-Test.png'))
    '''
    nComp = 10
    Xhat = np.dot(pca.transform(noise)[:,:nComp], pca.components_[:nComp,:])
    noise_level = np.dot(pca.transform(noise)[:,nComp:], pca.components_[nComp:,:])
    Xhat += mu
    SNR = np.linspace(1,5,50)
    for i  in range(len(SNR)):
        noise_test.append(SNR[i]*noise_level + Xhat)
        j = 0
        for spectra in noise_test[i]:
            noise_test[i][j] = spectra/np.max(spectra)
            j += 1
        noise_acc.append(model.evaluate(x = noise_test[i].reshape(474,500,1), y = y_test ))

    noise_acc = np.array(noise_acc)

    return SNR, noise_acc, noise_test

def plot_chemical_shift_test(model, model_MLP=None):
    if model_MLP != None:
        chemical_shift_test_MLP = chemical_shift_test(model_MLP)
    chemical_shift_test_CNN = chemical_shift_test(model)
    delta_E = np.linspace(-10.,10.,200)

    plt.figure(1, figsize=(10,6))
    if model_MLP != None:
        plt.plot(delta_E, chemical_shift_test_MLP[:,1], 'r', label='Vanilla Neural Network', linewidth = 2)
    plt.plot(delta_E, chemical_shift_test_CNN[:,1], 'b', label='Convolutional Neural Network', linewidth = 2)
    plt.axvline(x=0, c = 'black')
    plt.ylabel('Accuracy')
    plt.xlabel('Chemical Shift (eV)')
    plt.legend(loc = 'lower left')
    plt.savefig(os.path.join(path_to_output, 'chemical_shift_test.png'))

def chemical_shift_test(model):
    argmax = [model.evaluate(x = X_test[:,0+i:500+i], y = y_test ) for i in range(200)]
    argmax = np.array(argmax)
    return argmax

if __name__ == "__main__":
    test_model(sys.argv)
