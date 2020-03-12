import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import svm, grid_search
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn import metrics

#import tensorflow as tf
#tf.python.control_flow_ops = tf
from sklearn.cross_validation import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import gzip
import pandas as pd
import pdb
import random
from random import randint
import scipy.io

from keras.layers import Input, Dense
from keras.engine.training import Model
from keras.models import Sequential, model_from_config
from keras.layers.core import  Dropout, Activation, Flatten, Merge
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adam
from keras.layers import normalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras import regularizers
from keras.constraints import maxnorm



def multiple_layer_autoencoder(X_train, X_test, activation = 'linear', batch_size = 100, nb_epoch = 100, last_dim = 64):
    nb_hidden_layers = [X_train.shape[1], 256, 128, last_dim]
    X_train_tmp = np.copy(X_train)
    #X_test_tmp = np.copy(X_test)
    encoders = []
    autoencoders = []
    for i, (n_in, n_out) in enumerate(zip(nb_hidden_layers[:-1], nb_hidden_layers[1:]), start=1):
        print('Training the layer {}: Input {} -> Output {}'.format(i, n_in, n_out))
        # Create AE and training
        #ae = Sequential()
	input_img = Input(shape=(n_in,))
        #encoder = containers.Sequential([Dense(input_dim=n_in, output_dim=n_out, activation=activation)])
        encode = Dense(output_dim=n_out, activation=activation)(input_img)
        #decoder = containers.Sequential([Dense(input_dim=n_out,output_dim=n_in, activation=activation)]) #, activation=activation)])
        decode = Dense(output_dim=n_in, activation=activation)(encode) #, activation=activation)])
        #ae.add(AutoEncoder(encoder=encoder, decoder=decoder,
        #                   output_reconstruction=True))
        #ae1 = AutoEncoder(encoder=encoder, decoder=decoder,output_reconstruction=True)
	autoencoder = Model(input=input_img, output=decode)
	encoder = Model(input=input_img, output=encode)
	encoded_input = Input(shape=(n_out,))
	decoder_layer = autoencoder.layers[-1]
	decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
	autoencoder.compile(loss='mean_squared_error', optimizer='adam')
	autoencoder.fit(X_train_tmp, X_train_tmp,nb_epoch=10,batch_size=100 ,shuffle=True,validation_data=None, verbose=1)
	encoder.compile(loss='mean_squared_error', optimizer='adam')
	#ae.add(encoder)
	#ae.add(decoder)
	#ae.add(ae1)
        #ae.add(Dropout(0.5))
        #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        #ae.compile(loss='mean_squared_error', optimizer='adam')#  adam  'rmsprop')
        #ae.fit(X_train_tmp, X_train_tmp, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=None,show_accuracy=True, verbose=1)
        # Store trainined weight and update training data
        #encoders.append(ae.layers[0].encoder)
        encoders.append(encoder)
	autoencoders.append(autoencoder)
	#ae1.output_reconstruction= False
        X_train_tmp = encoder.predict(X_train_tmp,batch_size = 100)
        print X_train_tmp.shape
        #X_test_tmp = ae.predict(X_test_tmp)
        
    #return encoders, X_train_tmp, X_test_tmp
    return encoders,autoencoders

def autoencoder_two_subnetwork_fine_tuning(X_train1, X_train2, Y_train, X_test1, X_test2, Y_test = None, batch_size =100, nb_epoch = 10):
    print 'autoencode learning'
    last_dim = 64
    encoders1, autoencoders1 = multiple_layer_autoencoder(X_train1, X_test1, activation = 'sigmoid', batch_size = batch_size, nb_epoch = nb_epoch, last_dim = last_dim)
    encoders2, autoencoders2 = multiple_layer_autoencoder(X_train2, X_test2, activation = 'sigmoid', batch_size = batch_size, nb_epoch = nb_epoch, last_dim = last_dim)
    #pdb.set_trace()
    
    X_train1_tmp_bef = np.copy(X_train1)
    X_test1_tmp_bef = np.copy(X_test1) 
    for ae in encoders1:
        X_train1_tmp_bef = ae.predict(X_train1_tmp_bef)
        print X_train1_tmp_bef.shape
        X_test1_tmp_bef = ae.predict(X_test1_tmp_bef)
    
    X_train2_tmp_bef = np.copy(X_train2)
    X_test2_tmp_bef = np.copy(X_test2) 
    for ae in encoders2:
        X_train2_tmp_bef = ae.predict(X_train2_tmp_bef)
        print X_train2_tmp_bef.shape
        X_test2_tmp_bef = ae.predict(X_test2_tmp_bef)
        
    prefilter_train_bef = np.concatenate((X_train1_tmp_bef, X_train2_tmp_bef), axis = 1)
    prefilter_test_bef = np.concatenate((X_test1_tmp_bef, X_test2_tmp_bef), axis = 1)
        
    
    return prefilter_train_bef, prefilter_test_bef