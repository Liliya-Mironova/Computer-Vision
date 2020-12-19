import joblib
import random

import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from skimage.feature import hog
from pathlib import Path

from mnist import MNIST
from skimage import io
from sudoku import *
from skimage.color.colorconv import rgb2gray

# download 4 MNIST files you can here:
# http://yann.lecun.com/exdb/mnist/


MNIST_CELL_SIZE = 28
SEED = 666


def fix_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def train(save_model_path, mnist_data_path='mnist_data', print_accuracy=False):
    
    # Dear course team,
    #
    # In this homework I used 2 models:
    # the first one is a Random Forest, trained on MNIST. It predicts the 
    # probabilities for each digit in sudoku square.
    # Then, the second model - another Random Forest - is trained using those probabilities as the features and 
    # the right digits as the answers, on all train images.
    
    # The first model:
    fix_seed(SEED)
    
    mnist_data = MNIST(mnist_data_path)
    mnist_data.gz = True

    images_train, labels_train = mnist_data.load_training()
    images_test, labels_test = mnist_data.load_testing()

    images_train = np.uint8([np.reshape(im, (MNIST_CELL_SIZE,) * 2) for im in images_train])
    images_test = np.uint8([np.reshape(im, (MNIST_CELL_SIZE,) * 2) for im in images_test])
    labels_train, labels_test = np.int16(labels_train), np.int16(labels_test)

    hhog = lambda image: hog(image, orientations=8, pixels_per_cell=(8, 8),
                        cells_per_block=(3, 3), visualize=False, multichannel=False)

    features_train = np.array([hhog(im) for im in images_train])
    features_test = np.array([hhog(im) for im in images_test])
    # features_train = np.array([im.ravel() for im in images_train])
    # features_test = np.array([im.ravel() for im in images_test])

#     rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)
    rf = XGBClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)
    rf.fit(features_train, labels_train)

    model_fname = 'random_forest.joblib'
    joblib.dump(rf, Path(save_model_path) / model_fname)


    print_accuracy = True

    if print_accuracy:
        from sklearn.metrics import accuracy_score
        print(accuracy_score(labels_test, rf.predict(features_test)))

        
    # The second model:
    prefix = '/autograder/source/train/'
#     prefix = ''
    
    images_arr = [prefix + 'train_0.jpg',
              prefix + 'train_2.jpg',
              prefix + 'train_3.jpg',
              prefix + 'train_4.jpg',
              prefix + 'train_6.jpg',
              prefix + 'train_7.jpg',
              prefix + 'train_8.jpg']
    
    truth_arr = [[5, 3, 6, 5, 1, 4, 3, 4, 8, 7, 2, 6, 8, 6, 2, 9, 5, 9, 3, 4, 2, 9, 5, 6, 8, 1, 5, 6, 1, 8, 4, 7, 5], # 0
             [1, 7, 8, 6, 5, 1, 2, 6, 5, 2, 7, 3, 5, 9, 2, 8, 1, 4, 9, 7, 1, 6, 8, 4, 7, 1, 5, 6, 9, 7, 6], # 2.1
             [8, 3, 4, 8, 9, 3, 7, 4, 2, 3, 4, 7, 5, 7, 6, 4, 7, 8, 6, 9, 5, 4, 1, 5, 9, 5, 1, 6, 5, 9, 4, 8, 3], # 2.2
             [2, 9, 3, 5, 7, 4, 9, 5, 6, 2, 9, 3, 2, 1, 8, 1, 9, 4, 6, 3, 5, 2, 4, 3, 7, 6, 8, 3, 7, 4, 1, 9], # 3
             [6, 5, 1, 6, 5, 8, 3, 5, 9, 4, 3, 6, 2, 1, 5, 7, 3, 9, 1, 5, 5, 8, 6, 5, 9, 4, 2, 1, 5, 8, 6, 1, 6, 2, 5], # 4
             [7, 5, 9, 1, 3, 4, 2, 3, 8, 5, 2, 9, 3, 6, 7, 2, 9, 8, 1, 2, 4, 6, 3, 3, 6, 5, 3, 1], # 6
             [9, 7, 8, 1, 1, 9, 8, 6, 5, 7, 2, 5, 3, 9, 7, 8, 6, 2, 6, 1, 9, 7, 8, 4, 1, 5, 2, 4, 8, 7, 2, 9, 3, 9, 5, 4, 7, 2, 1, 7, 4], # 7
             [4, 5, 7, 6, 5, 6, 3, 2, 1, 5, 1, 9, 8, 6, 7, 4, 8, 4, 3, 6, 6, 4, 7, 1, 2, 8, 3, 1, 5, 4, 7, 1, 8, 3, 4, 7]] # 8

    truths, pred_probas = [], []
    
    for truth in truth_arr:
        truths.append(truth)
    truths = np.hstack(truths)
    
    for i, fname in enumerate(images_arr):
        image = cv2.imread(fname)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = io.imread(fname, as_gray=True)
        image_full = image
    
        mask, corners = mask_image(image_full)
        images_warped, tforms = normalize_image(image_full)

        sudoku_digits = []
        for image_warped in images_warped:
            pred_square, pred_proba, flag, cells, _ = detect_digits(image_warped, rf, get_templates=False)
            pred_probas.append(pred_proba)
            
    pred_probas = np.vstack(pred_probas)
   
#     rf2 = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=666)
    rf2 = XGBClassifier(n_estimators=50, n_jobs=-1, random_state=666)
    rf2.fit(pred_probas, truths)
    
    model_fname = 'random_forest2.joblib'
    joblib.dump(rf2, Path(save_model_path) / model_fname)