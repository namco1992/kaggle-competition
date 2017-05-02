
# coding: utf-8

# In[1]:

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json

import cv2

# import tensorflow
# import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

from tqdm import tqdm

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.metrics import fbeta_score
import time


_BATCH_SIZE = 32
_EPOCH = 50
_IMG_SIZE = (64, 64)
_INPUT_SHAPE = (64, 64, 3)
_NUM_OF_CLASSES = 17
_NFOLDS = 5
_LABELS = [
    'haze',
    'artisinal_mine',
    'blooming',
    'habitation',
    'cultivation',
    'primary',
    'clear',
    'water',
    'road',
    'slash_burn',
    'cloudy',
    'agriculture',
    'partly_cloudy',
    'conventional_mine',
    'bare_ground',
    'selective_logging',
    'blow_down']


# def read_csv(path):
#     df_train = pd.read_csv('../input/train.csv')
#     df_test = pd.read_csv('../input/sample_submission.csv')

def init_labels():
    label_map = {l: i for i, l in enumerate(_LABELS)}
    inv_label_map = {i: l for l, i in label_map.items()}
    return label_map, inv_label_map


def load_datasets():
    df_train = pd.read_csv('../input/train.csv')
    df_test = pd.read_csv('../input/sample_submission.csv')

    x_train = []
    x_test = []
    y_train = []

    label_map, inv_label_map = init_labels()
    for f, tags in tqdm(df_train.values, miniters=1000, ascii=True):
        img = cv2.imread('../input/train-jpg/{}.jpg'.format(f))
        targets = np.zeros(17)
        for t in tags.split(' '):
            targets[label_map[t]] = 1
        x_train.append(cv2.resize(img, _IMG_SIZE))
        y_train.append(targets)

    for f, tags in tqdm(df_test.values, miniters=1000, ascii=True):
        img = cv2.imread('../input/test-jpg/{}.jpg'.format(f))
        x_test.append(cv2.resize(img, _IMG_SIZE))

    y_train = np.array(y_train, np.uint8)
    x_train = np.array(x_train, np.float32) / 255.
    x_test = np.array(x_test, np.float32) / 255.

    print(x_train.shape)
    print(y_train.shape)

    return x_train, y_train, x_test


def load_model(weights_path=None):
    model = Sequential()

#     model.add(Conv2D(32, 3, 3, activation='relu', input_shape=(48, 48, 3)))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Conv2D(48, 3, 3, activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#     model.add(Conv2D(64, 3, 3, activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(17, activation='softmax'))

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=_INPUT_SHAPE))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(_NUM_OF_CLASSES))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    if weights_path:
        model.load_weights(weights_path)
    return model


# In[11]:

def train(x_train, y_train, x_test, load_weights=False):
    num_fold = 0
    # sum_score = 0
    # yfull_test = []
    # yfull_train =[]
    historys = []
    f_scores = []

    kf = KFold(len(y_train), n_folds=_NFOLDS, shuffle=True, random_state=1)

    for train_index, test_index in kf:
        start_time_model_fitting = time.time()
        # print(train_index, test_index)
        X_train = x_train[train_index]
        Y_train = y_train[train_index]
        X_valid = x_train[test_index]
        Y_valid = y_train[test_index]

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, _NFOLDS))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))

        kfold_weights_path = os.path.join('', 'weights_kfold_' + str(num_fold) + '.h5')
        if load_weights is True:
            model = load_model(kfold_weights_path)
        else:
            model = load_model()

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2, verbose=0),
            ModelCheckpoint(
                kfold_weights_path, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=0)]

        history = model.fit(
            x=X_train, y=Y_train, validation_data=(X_valid, Y_valid),
            batch_size=_BATCH_SIZE, verbose=2, epochs=_EPOCH, callbacks=callbacks,
            shuffle=True)
        historys.append(history.history)

        if os.path.isfile(kfold_weights_path):
            model.load_weights(kfold_weights_path)

        # p_valid = model.predict(X_valid, batch_size=128, verbose=2)
        p_valid = predict(model, X_valid)

        score = evaluate(Y_valid, p_valid)
        # f_scores_dict = {}
        # for i in range(5, 20):
        #     threshold = (i + 1) / 100.
        #     score = fbeta_score(Y_valid, np.array(p_valid) > threshold, beta=2, average='samples')
        #     f_scores_dict[threshold] = score
        f_scores.append(score)

    end_time = time.time()
    with open('history_{}.json'.format(int(end_time)), 'w') as history_file:
        history_file.write(json.dumps(historys))

    with open('fscore_{}.json'.format(int(end_time)), 'w') as fscore_file:
        fscore_file.write(json.dumps(f_scores))


    #         p_test = model.predict(x_train, batch_size = 128, verbose=2)
    #         yfull_train.append(p_test)


def predict(model, x, batch_size=128, verbose=2):
    p = model.predict(x, batch_size=batch_size, verbose=verbose)
    return p


def evaluate(y_true, y_pred, metrics='fbeta_score'):
    fbeta_scores_dict = {}
    for i in range(5, 20):
        threshold = (i + 1) / 100.
        score = fbeta_score(y_true, np.array(y_pred) > threshold, beta=2, average='samples')
        fbeta_scores_dict[threshold] = score
    return fbeta_scores_dict


# import matplotlib.pyplot as plt

# plot_num = len(historys)
# for i in range(plot_num):
#     history_dict = historys[i].history

#     # history_dict = history.history
#     loss_values = history_dict['loss']
#     val_loss_values = history_dict['val_loss']
#     epochs = range(1, len(loss_values) + 1)
#     plt.subplot(plot_num, 1, i + 1)
#     # "bo" is for "blue dot"
#     plt.plot(epochs, loss_values, 'bo')
#     # b+ is for "blue crosses"
#     plt.plot(epochs, val_loss_values, 'b+')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')

#     plt.show()


# # In[ ]:

# for i in range(plot_num):
#     history_dict = historys[i].history
#     acc_values = history_dict['acc']
#     val_acc_values = history_dict['val_acc']
# #     print(acc_values)
#     epochs = range(1, len(loss_values) + 1)

#     plt.subplot(plot_num, 1, i+1)

#     plt.plot(epochs, acc_values, 'bo')
#     plt.plot(epochs, val_acc_values, 'b+')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')

#     plt.show()

if __name__ == '__main__':
    x_train, y_train, x_test = load_datasets()
    train(x_train, y_train, x_test)
