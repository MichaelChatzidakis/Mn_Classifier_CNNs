import glob
from keras.models import load_model
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import glob
import os
import sys
from train_crossval import load_data
from sklearn.model_selection import StratifiedKFold
from keras.utils import np_utils

def main(argv):
    data_path = '/home/mike/Mn_Valences/Mn_Classifier_CV_Good_Copy/Data/Digitized_Mn_Usecases.pkl'
    data = pd.read_pickle(data_path)

    valence_names = ['Mn4', 'Mn2', 'Mn3', 'mixed23', 'mixed34', 'mixed']
    spectra_set,key=[],[]
    for valence in range(len(data)):
        for spectra in range(len(data[valence])):
            bins = 300
            lower_bound=600
            upper_bound=700
            x = data[valence].iloc[spectra]['Energy']
            y = data[valence].iloc[spectra]['Intensity']
            x = x[(x>lower_bound)]
            x = x[(x<upper_bound)]
            y = y[x.index]

            min_energy = np.min(x)
            max_energy = np.max(x)
            new_energy=np.linspace(min_energy,max_energy,bins)
            new_intensity = np.interp(new_energy, x, y)
            spectra_set.append(new_intensity)
            key.append(spectra)

    labels = [2]*12+[0]*10+[1]*9
    spectra_set=spectra_set[:len(labels)]

    x,y = load_data()
    cv = StratifiedKFold(n_splits=10, random_state=13, shuffle=False)
    X_train = [x[train_index] for train_index, test_index in cv.split(x, y)]
    X_test = [x[test_index] for train_index, test_index in cv.split(x, y)]
    y_train = [y[train_index] for train_index, test_index in cv.split(x, y)]
    y_test = [y[test_index] for train_index, test_index in cv.split(x, y)]


    spectra_set=np.array(spectra_set).astype('float32')
    spectra_set -= np.mean(x)
    spectra_set /= np.max(x)
    spectra_set = spectra_set.reshape(spectra_set.shape + (1,))
    labels = np.array(labels)
    labels = np_utils.to_categorical(labels)

    neural_network_name=['mlp_500epochs_Shift_dataaug-x0',
                         'cnn_500epochs_Shift_dataaug-x0',
                         'cnnmlp_500epochs_Shift_dataaug-x0',
                         'mlp_500epochs_Shift_dataaug-x1',
                         'cnn_500epochs_Shift_dataaug-x1',
                         'cnnmlp_500epochs_Shift_dataaug-x1',
                         'mlp_500epochs_Shift_dataaug-x10',
                         'cnn_500epochs_Shift_dataaug-x10',
                         'cnnmlp_500epochs_Shift_dataaug-x10']

    scores = np.zeros((9,10))
    for j,name in enumerate(neural_network_name):
        print(name)
        weights_paths=load_best_weights(name)
        for i in range(10):
            print("fold", i)
            model=load_model(weights_paths[i])
            scores[j][i]=model.evaluate(spectra_set, labels, verbose=0)[1]
            print(scores[j][i])
    import pdb; pdb.set_trace()
    pd.DataFrame(scores).to_csv('{}_scores_dig_ref_spec.csv'.format(neural_network_name))

def load_best_weights(model):
    root_path = os.path.join('/home/mike/Mn_Valences/Mn_Classifier_Reviewer_edits/weights/cross_validation_results/', model)
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

def get_pred_score(spectra_set, models):
    pred_score_mlp1=[]
    for j, unknown_spectra in enumerate(spectra_set):
        pred_label=[]
        print("loading spectra: {}.".format( round(unknown_spectra.mean(),4) ) )
        for i in range(10):
            print(j, i)
            data_set = X_train[i]
            unknown_spectra -=  np.mean(data_set)
            unknown_spectra /= np.max(data_set)
            pred = models[i].predict(unknown_spectra.reshape(1,300,1), verbosity=1)
            pred_label.append(np.argmax(pred))
        pred_label=pred_label
        pred_score_mlp1.append(pred_label)
        print("Real label is {}, predicted label is {}".format(labels[j], pred_score_mlp1[j]))
    return pred_score_mlp1

def get_acc_scores(pred_score_cnnmlp):
    ss=pd.DataFrame([(pred_score_cnnmlp)]).transpose()
    ss['round'] =np.round(pred_score_cnnmlp).astype('int')
    ss['labels']=labels
    ss['correct'] = ss['round']==ss['labels']
    acc=[]
    for valence in [0,1,2]:
        acc.append(ss[ss['labels']==valence]['correct'].sum()/len( ss[ss['labels']==valence]['correct'])*1.0)
    return acc

if __name__ == "__main__":
    main(sys.argv)
