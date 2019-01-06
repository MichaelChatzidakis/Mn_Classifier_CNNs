import os
import sys
import glob
import numpy as np
np.random.seed(23087)
import pandas as pd
from keras import backend as K
from keras.utils import np_utils
from keras.models import load_model
from sklearn.metrics import log_loss
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from train_crossval import preprocess_crossval_aug, load_data
from sklearn.metrics import confusion_matrix

def test_model(argv):
      #Load Data
    path_to_output = 'figures'
    Mn_All,labels = load_data()
    weight = load_best_weights(model=argv[2])
    num_bins_translate = 100
    num_classes = 3

    total_confusion_matrix=np.zeros((num_classes,num_classes))
    acc = np.zeros(num_bins_translate)
    noise_acc = np.zeros(50)
    chemical_shift_test_acc, noise_test_acc = [], []
    if argv[1] == 'single':
        for fold in range(len(weight)):
            model = load_model(weight[fold])
            print ('Only one model is being tested.')
            #we need the uncropped spectra for the chemical shift test
            (X_train, y_train), (X_test, y_test) = preprocess_crossval_aug(Mn_All, labels, cv=True, fold=fold, n_splits=len(weight), crop_spectra=False, aug = False)
            run = chemical_shift_test(X_test, y_test, model, num_bins_translate)[:,1]
            acc += run
            print( run)
            #SNR,noise_acc, noise_test = noise_PCA_spectra(model, X_test, y_test)
            #noise_acc += noise_acc
            #(X_train, y_train), (X_test, y_test) = preprocess_crossval_aug(Mn_All, labels, cv=True, fold=fold, n_splits=len(weight), crop_spectra=False, aug = False)
            total_confusion_matrix += confusion_matrix_generator(X_test[:,200:500], y_test, model)
            #print(noise_acc)

        #noise_test_acc.append(noise_acc[:,1]/len(weight))
        chemical_shift_test_acc.append(acc/len(weight))

    elif argv[1] == 'compare':
        weights = [load_best_weights(model=argv[2]), load_best_weights(model=argv[3])]
        for weight in weights:
            acc = np.zeros(num_bins_translate)
            noise_acc = np.zeros(50)
            total_confusion_matrix=np.zeros((num_classes,num_classes))
            for fold in range(len(weights[0])):
                model = load_model(weight[fold])

                #we need the uncropped spectra for the chemical shift test
                (X_train, y_train), (X_test, y_test) = preprocess_crossval_aug(Mn_All, labels, cv=True, fold=fold, n_splits=len(weight), crop_spectra=False, aug = True)
                acc+=chemical_shift_test(X_test, y_test, model, num_bins_translate)[:,1]
                SNR,noise_acc, noise_test = noise_PCA_spectra(model, X_test, y_test)
                noise_acc += noise_acc

                (X_train, y_train), (X_test, y_test) = preprocess_crossval_aug(Mn_All, labels, cv=True, fold=fold, n_splits=len(weight), crop_spectra=True, aug = True)
                total_confusion_matrix += confusion_matrix_generator(X_test, y_test, model)
                #plot_noise_test(model, spectra= 451)

            noise_test_acc.append(noise_acc[:,1]/len(weight[0]))
            chemical_shift_test_acc.append(acc/len(weight[0]))
            total_cv_confusion_matrix_generator(total_confusion_matrix)

    else:
        print("Specify if results are for one model or comparing models.")

    for i in range(len(chemical_shift_test_acc)):
        plot_chemical_shift_test(path_to_output, num_bins_translate, chemical_shift_test_acc[i])
        #plot_noise_test(path_to_output, SNR, noise_test, 7, noise_test_acc[i])
    total_cv_confusion_matrix_generator(total_confusion_matrix)

def load_best_weights(model):
    root_path = os.path.join('weights','cross_validation_results', model)

    try:
        weight_folds=sorted(next(os.walk(root_path))[1])
    except StopIteration:
        pass

    weights=[]
    for fold in weight_folds:
        files_path = os.path.join(root_path, fold, '*.h5')
        cv_weights = sorted(glob.iglob(files_path), key=os.path.getctime, reverse=True)
        weights.append(cv_weights[0])
    return weights

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

def plot_noise_test(path_to_output, SNR, noise_test, spectra, noise_test_acc, noise_test_acc2=None):

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
    plt.plot(SNR, noise_test_acc, '.', markersize=15)
    if type(noise_test_acc2) == np.ndarray:
        plt.plot(SNR, noise_test_acc, '.', markersize=15)
    plt.ylabel('Accuracy')
    plt.xlim(0.95, 5.05)
    plt.xticks([1,2,3,4,5])
    plt.yticks([0.60, 0.70, 0.80, 0.90, 1.0])
    plt.xlabel('Multiples of Noise Added')
    plt.savefig(os.path.join(path_to_output, 'accuracy_vs_noise.png'))

def noise_PCA_spectra(model, X_test, y_test,scree_plot=None):
    noise_test, noise_acc = [], []
    noise = np.copy(X_test[:,100:600,0])
    mu = np.mean(noise, axis=0)

    pca = PCA()
    noise_model = pca.fit(noise)
    if scree_plot == True:
        plt.figure(5)
        plt.title('Scree Plot')
        plt.plot(pca.explained_variance_ratio_, 'r.')
        plt.xlim(-0.1,30)
        plt.ylim(-0.01, 0.6)
        plt.xlabel('Principal Components')
        plt.savefig(os.path.join(path_to_output, 'Scree-Plot-for-Noise-Test.png'))

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
        noise_acc.append(model.evaluate(x = noise_test[i].reshape(len(X_test),500,1), y = y_test, verbose = 0 ))

    noise_acc = np.array(noise_acc)

    return SNR, noise_acc, noise_test

def plot_chemical_shift_test(path_to_output, num_bins_translate, acc1, acc2=None):
    delta_E = np.linspace(-5.,5.,100)
    plt.figure(1, figsize=(10,6))
    plt.plot(delta_E, acc1, 'b', label='Dense Network with Shift Aug.', linewidth = 2)
    plt.axvline(x=0, c = 'black')
    plt.ylabel('10-fold Cross Validation Test Accuracy')
    plt.xlabel('Chemical Shift (eV)')
    plt.legend(loc = 'lower left')
    plt.savefig(os.path.join(path_to_output, 'chemical_shift_test.png'))
    plt.show()

def chemical_shift_test(X_test, y_test, model, num_bins_translate):

    accuracy_on_shifted_test_set = [model.evaluate(x = X_test[:,150+i:450+i], y = y_test, verbose=0 ) for i in range(num_bins_translate)]
    import pdb; pdb.set_trace()
    accuracy_on_shifted_test_set = np.array(accuracy_on_shifted_test_set)
    return accuracy_on_shifted_test_set

if __name__ == "__main__":
    test_model(sys.argv)
