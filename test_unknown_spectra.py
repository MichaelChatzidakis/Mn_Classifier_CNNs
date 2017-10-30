import sys
import os
import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from keras.models import load_model
from scipy import optimize, misc

crop_min = 600.
crop_max = 680.
dispersion = 0.4

spectra_path = '/home/mike/Mn_Valences/Test_Spectra/NMC_531_Z1/ICA-Triple-GB_Segr.msa'
model = load_model('/home/mike/Mn_Valences/Mn_Classifier_Good_Copy/weights/CNN_Noise_DataAug/highest_val_acc_weights_epoch98-val_acc0.994_cnn.h5')

def test_spectra(argv):
    unknown_spectra = load_spectra(spectra_path)

    unknown_spectra = subtract_background(unknown_spectra, crop_min, crop_max, indices_before_onset=60)

    resampled_spectra = preprocess_spectra(unknown_spectra, 620, 660, crop_min, crop_max, dispersion)

    predict_spectra(resampled_spectra)

def load_spectra(spectra_path):
    s = hs.load(spectra_path)
    s.crop(-1,crop_min, crop_max)
    s = np.array(s)
    return s

def PowerLaw(p, x): #substract background (powerlaw)
    return p[0]*x**(p[1])

def subtract_background(unknown_spectra, crop_min, crop_max, indices_before_onset=60):
    x = np.linspace(crop_min, crop_max, 200)
    errfunc = lambda p, x, unknown_spectra : PowerLaw(p, x) -  unknown_spectra  # Distance to the target function
    p0 = [ -0.4,  0.15]
    p1, success = optimize.leastsq(errfunc, p0, args=(x[0:indices_before_onset], ( np.array(unknown_spectra[0:indices_before_onset]) )), maxfev=500000)
    unknown_spectra -= PowerLaw(p1, x)
    return unknown_spectra

def preprocess_spectra(unknown_spectra, min, max, crop_min, crop_max, dispersion):
    unknown_spectra = (unknown_spectra[(int((min-crop_min)/dispersion)):(len(unknown_spectra) - (int((crop_max-max)/dispersion)))])
    unknown_spectra = np.array(unknown_spectra).astype('float32')
    unknown_spectra = unknown_spectra.reshape(unknown_spectra.shape + (1,))
    unknown_spectra -=  np.mean(unknown_spectra)
    unknown_spectra /= np.max(unknown_spectra)
    f = signal.resample(unknown_spectra, 500)
    return f

def predict_spectra(f):
    label = ['Mn2+', 'Mn3+', 'Mn4+']
    pred = model.predict(f.reshape(1,500,1))
    print("Class probabilities: ", pred)
    print("The predicted valence is: ", label[np.argmax(pred)])
if __name__ == "__main__":
    test_spectra(sys.argv)
