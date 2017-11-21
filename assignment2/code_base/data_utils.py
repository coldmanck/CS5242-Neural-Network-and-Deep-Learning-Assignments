from __future__ import print_function

from builtins import range
from six.moves import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread
import platform

def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000,
                     subtract_mean=True):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'code_base/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }

def get_CIFAR2_data(num_training=9800, num_validation=200, num_test=2000,
                     subtract_mean=True):
    """
    Load the CIFAR-2 (class0: airplane, class2: bird) dataset from disk and perform preprocessing to prepare
    it for classifiers. 
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'code_base/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    
    # Find class0: airplane, class2: bird (will have total 5000 train (to split to 4900 train, 100 val), 1000 test each)
    class0_Xtrain = X_train[np.where( y_train == 0 )]  
    class2_Xtrain = X_train[np.where( y_train == 2 )]    
    class0_Xtest = X_test[np.where( y_test == 0 )]  
    class2_Xtest = X_test[np.where( y_test == 2 )] 
    
    # collate and shuffle
    X_train = np.append(class0_Xtrain, class2_Xtrain, axis=0)
    y_train = np.append(np.zeros(class0_Xtrain.shape[0], dtype='int64'), 
                        np.ones(class2_Xtrain.shape[0], dtype='int64'), axis=0)
    
    X_test = np.append(class0_Xtest, class2_Xtest, axis=0)
    y_test = np.append(np.zeros(class0_Xtest.shape[0], dtype='int64'), 
                        np.ones(class2_Xtest.shape[0], dtype='int64'), axis=0)
    #shuffle train and test data
    mask = list(range(X_train.shape[0]))
    np.random.shuffle(mask)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(X_test.shape[0]))
    np.random.shuffle(mask)
    X_test = X_test[mask]
    y_test = y_test[mask]
    
    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    np.random.shuffle(mask)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    np.random.shuffle(mask)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    np.random.shuffle(mask)
    X_test = X_test[mask]
    y_test = y_test[mask]
    
    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }



def load_models(models_dir):
    """
    Load saved models from disk. This will attempt to unpickle all files in a
    directory; any files that give errors on unpickling (such as README.txt)
    will be skipped.

    Inputs:
    - models_dir: String giving the path to a directory containing model files.
      Each model file is a pickled dictionary with a 'model' field.

    Returns:
    A dictionary mapping model file names to models.
    """
    models = {}
    for model_file in os.listdir(models_dir):
        with open(os.path.join(models_dir, model_file), 'rb') as f:
            try:
                models[model_file] = load_pickle(f)['model']
            except pickle.UnpicklingError:
                continue
    return models

