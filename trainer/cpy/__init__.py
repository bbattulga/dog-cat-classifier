from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import math
import os

dirname = os.path.dirname(__file__)


def relative(d):
    return os.path.join(dirname, d)


model = keras.Sequential([
    layers.Conv2D(32, (3, 3), input_shape=(250, 250, 3)),
    layers.Activation('relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(32, (3, 3)),
    layers.Activation('relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3)),
    layers.Activation('relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(64),
    layers.Activation('relu'),
    layers.Dropout(0.5),
    layers.Dense(1),
    layers.Activation('sigmoid')
])
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
img_width, img_height = 250, 250
batch_size = 1
nb_train_samples = len(os.listdir(relative('train/cat')) +
                       os.listdir(relative('train/dog')))
nb_validation_samples = len(os.listdir(
    relative('val/cat')) + os.listdir(relative('val/dog')))
nb_test_samples = len(os.listdir(relative('test/cat')) +
                      os.listdir(relative('test/dog')))


data_gen = ImageDataGenerator(rescale=1./255)
train_generator = data_gen.flow_from_directory(
    relative('train'), batch_size=batch_size, target_size=(img_width, img_height), class_mode='binary')

val_generator = data_gen.flow_from_directory(
    relative('val'), batch_size=batch_size, target_size=(img_width, img_height), class_mode='binary')

test_generator = data_gen.flow_from_directory(
    relative('val'), batch_size=batch_size, target_size=(img_width, img_height), class_mode='binary')

epochs = 10
history = model.fit_generator(train_generator,
                              steps_per_epoch=nb_train_samples//epochs,
                              validation_data=val_generator, epochs=epochs,
                              validation_steps=nb_validation_samples // epochs)
