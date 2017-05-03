
# coding: utf-8

# In[1]:

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
import logging
import time

import cv2

# import tensorflow
# import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from tqdm import tqdm

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.metrics import fbeta_score

from lsuv_init import LSUVinit

_BATCH_SIZE = 32
_EPOCH = 50
_IMG_SIZE = (128, 128)
_INPUT_SHAPE = (128, 128, 3)
_NUM_OF_CLASSES = 17
_NFOLDS = 10
_BN_AXIS = 3
_TRAIN_PATH = '../input/train-jpg/{}.jpg'
_TEST_PATH = '../input/test-jpg/{}.jpg'
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
        img = cv2.imread(_TRAIN_PATH.format(f))
        targets = np.zeros(17)
        for t in tags.split(' '):
            targets[label_map[t]] = 1
        x_train.append(cv2.resize(img, _IMG_SIZE))
        y_train.append(targets)

    # for f, tags in tqdm(df_test.values, miniters=1000, ascii=True):
    #     img = cv2.imread(_TEST_PATH.format(f))
    #     x_test.append(cv2.resize(img, _IMG_SIZE))

    y_train = np.array(y_train, np.uint8)
    x_train = np.array(x_train, np.float32) / 255.
    # x_test = np.array(x_test, np.float32) / 255.

    logging.info(x_train.shape)
    logging.info(y_train.shape)

    return x_train, y_train, x_test


def load_model(weights_path=None):
    model = Sequential()

    # Block 1
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=_INPUT_SHAPE, name='block1_conv1'))
    model.add(BatchNormalization(axis=_BN_AXIS, name='block1_bn1'))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), name='block1_conv2'))
    model.add(BatchNormalization(axis=_BN_AXIS, name='block1_bn2'))
    model.add(Activation('relu'))

    model.add(Conv2D(128, (3, 3), name='block1_conv3'))
    model.add(BatchNormalization(axis=_BN_AXIS, name='block1_bn3'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Block 2
    model.add(Conv2D(64, (3, 3), name='block2_conv1'))
    model.add(BatchNormalization(axis=_BN_AXIS, name='block2_bn1'))
    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3), name='block2_conv2', padding='same'))
    model.add(BatchNormalization(axis=_BN_AXIS, name='block2_bn2'))
    model.add(Activation('relu'))

    model.add(Conv2D(128, (3, 3), name='block2_conv3'))
    model.add(BatchNormalization(axis=_BN_AXIS, name='block2_bn3'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Block 3
    model.add(Conv2D(128, (3, 3), name='block3_conv1'))
    model.add(BatchNormalization(axis=_BN_AXIS, name='block3_bn1'))
    model.add(Activation('relu'))

    model.add(Conv2D(128, (3, 3), name='block3_conv2', padding='same'))
    model.add(BatchNormalization(axis=_BN_AXIS, name='block3_bn2'))
    model.add(Activation('relu'))

    model.add(Conv2D(256, (3, 3), name='block3_conv3'))
    model.add(BatchNormalization(axis=_BN_AXIS, name='block3_bn3'))
    model.add(Activation('relu'))

    # Top
    model.add(Conv2D(512, (3, 3), name='top_conv'))
    model.add(BatchNormalization(axis=_BN_AXIS, name='top_bn'))
    model.add(Activation('relu'))

    model.add(AveragePooling2D(pool_size=(7, 7), name='avg_pool_top'))

    model.add(Flatten())
    model.add(Dense(_NUM_OF_CLASSES))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    if weights_path:
        model.load_weights(weights_path)
    return model


# In[11]:

def train(x_train, y_train, x_test=None, load_weights=False):
    num_fold = 0
    # sum_score = 0
    # yfull_test = []
    # yfull_train =[]
    historys = []
    f_scores = []

    kf = KFold(len(y_train), n_folds=_NFOLDS, shuffle=True, random_state=1)

    for train_index, test_index in kf:
        start_time_model_fitting = time.time()
        # logging.info(train_index, test_index)
        X_train = x_train[train_index]
        Y_train = y_train[train_index]
        X_valid = x_train[test_index]
        Y_valid = y_train[test_index]

        num_fold += 1
        logging.info('Start KFold number {} from {}'.format(num_fold, _NFOLDS))
        logging.info('Split train: ', len(X_train), len(Y_train))
        logging.info('Split valid: ', len(X_valid), len(Y_valid))

        kfold_weights_path = os.path.join('test2/', 'weights_kfold_' + str(num_fold) + '.h5')
        if load_weights is True:
            model = load_model(kfold_weights_path)
        else:
            model = load_model()

        # LSUV init
        model = LSUVinit(model, X_train[:_BATCH_SIZE,:,:,:])

        callbacks = [
            ReduceLROnPlateau(monitor='val_loss', patience=3),
            TensorBoard(log_dir='./Graph2', histogram_freq=0, write_graph=True, write_images=True),
            EarlyStopping(monitor='val_loss', patience=5, verbose=0),
            ModelCheckpoint(
                kfold_weights_path, monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=0)]

        train_datagen = ImageDataGenerator(
            fill_mode='nearest',
            data_format=K.image_data_format(),
            # rescale=1. / 255,
            zoom_range=0.1,
            rotation_range=10,
            width_shift_range=0.05,
            height_shift_range=0.05,
            horizontal_flip=True,
            vertical_flip=True)

        history = model.fit_generator(
            train_datagen.flow(X_train, Y_train, batch=_BATCH_SIZE),
            steps_per_epoch=len(X_train) // _BATCH_SIZE,
            validation_data=(X_valid, Y_valid),
            batch_size=_BATCH_SIZE,
            epochs=_EPOCH,
            callbacks=callbacks
        )
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
    for i in range(1, 20):
        threshold = (i + 1) / 100.
        score = fbeta_score(y_true, np.array(y_pred) > threshold, beta=2, average='samples')
        fbeta_scores_dict[threshold] = score
    return fbeta_scores_dict


if __name__ == '__main__':
    x_train, y_train, x_test = load_datasets()
    train(x_train, y_train, x_test)
