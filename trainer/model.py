import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd

from . import hyperparams


def build_model1():
    model = keras.Sequential([

        # input layer
        layers.Conv2D(32, (3, 3), input_shape=(
            hyperparams.img_width, hyperparams.img_height, 3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2, 2),

        # conv layers
        layers.Conv2D(64, (3, 3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(256, (3, 3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(512, (3, 3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(1024, (3, 3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2, 2),

        # flatten
        layers.Flatten(),

        # dense layers
        layers.Dense(1024),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        layers.Dense(1024),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        # output
        layers.Dense(1),
        layers.Activation('sigmoid')
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return model


def build_model2():
    model = keras.Sequential([

        # input layer
        layers.Conv2D(32, (3, 3), input_shape=(
            hyperparams.img_width, hyperparams.img_height, 3)),
        layers.Activation('relu'),
        layers.MaxPooling2D(2, 2),

        # conv layers
        layers.Conv2D(64, (3, 3)),
        layers.Activation('relu'),
        layers.MaxPooling2D(2, 2),

        # flatten
        layers.Flatten(),

        # dense layers
        layers.Dense(128),
        layers.Activation('relu'),

        # output
        layers.Dense(1),
        layers.Activation('sigmoid')
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    return model
