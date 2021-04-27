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
from . import dataset
from . import hyperparams
from .args import args
from . import model

gen = dataset.get_generators()

nb_train_samples = len(os.listdir('dog-cat-mini/train/cat') +
                       os.listdir('dog-cat-mini/train/dog'))
nb_validation_samples = len(os.listdir(
    'dog-cat-mini/val/cat') + os.listdir('dog-cat-mini/val/dog'))
nb_test_samples = len(os.listdir('dog-cat-mini/test/cat') +
                      os.listdir('dog-cat-mini/test/dog'))

print(nb_train_samples)
print(nb_test_samples)
print(nb_validation_samples)

dog_cat_model = model.build_model()
history = dog_cat_model.fit_generator(gen.train_gen,
                                      steps_per_epoch=gen.train_size//hyperparams.epochs,
                                      validation_data=gen.val_gen, epochs=hyperparams.epochs,
                                      validation_steps=gen.val_size // hyperparams.epochs)
