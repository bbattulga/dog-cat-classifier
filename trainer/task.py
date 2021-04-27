import matplotlib.pyplot as plt
import time
import argparse
from tensorflow import keras
from tensorflow.python.lib.io import file_io
import tensorflow as tf
import os
import logging

from . import dataset
from . import model
from . import hyperparams

from .args import args

dataset_gen = dataset.get_generators()

dogcat_model = model.build_model2()
dogcat_model.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

epochs = hyperparams.epochs
batch_size = hyperparams.batch_size

history = dogcat_model.fit(
    dataset_gen.train_gen,
    epochs=epochs,
    validation_data=dataset_gen.val_gen)

dogcat_model.evaluate(dataset_gen.test_gen)

# save to gcp
gcp_save_root = f'{args.job_dir}'


def visualize(history_dict):
    global gcp_save_root
    # visualize history_dict
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history_dict['loss'], label='Training loss')
    plt.plot(history_dict['val_loss'], label='Validation loss')
    plt.legend()
    print('plotted loss')

    plt.subplot(1, 2, 2)
    plt.plot(history_dict['accuracy'], label='Training accuracy')
    plt.plot(history_dict['val_accuracy'], label='Validation accuracy')
    plt.legend()

    print('save')
    fig_path = './visual.png'
    plt.savefig(fig_path)
    gcp_visual_save_path = f'{gcp_save_root}/visual-small.png'
    with file_io.FileIO(fig_path, mode='rb') as saved_fig:
        with file_io.FileIO(gcp_visual_save_path, mode='wb+') as output:
            output.write(saved_fig.read())
            print(f'saved to {gcp_visual_save_path}')


visualize(history.history)

# save model
gcp_model_save_path = f'{gcp_save_root}/dog-cat-{time.time()}.h5'
model_save_path = './dog-cat.h5'
dogcat_model.save(model_save_path)
with file_io.FileIO(model_save_path, mode='rb') as saved_model:
    with file_io.FileIO(gcp_model_save_path, mode='wb+') as output:
        output.write(saved_model.read())
        print(f'saved to {gcp_model_save_path}')
