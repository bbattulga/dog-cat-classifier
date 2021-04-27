import os

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from zipfile import ZipFile
from google.cloud import storage

from . import hyperparams
from .args import args

dataset_name = 'dog-cat'


class DatasetGenerator:

    def __init__(self, train_gen=None, val_gen=None, test_gen=None, train_size=None, test_size=None, val_size=None):
        self.train_gen = train_gen
        self.train_size = train_size

        self.test_gen = test_gen
        self.test_size = test_size

        self.val_gen = val_gen
        self.val_size = val_size


def get_generators():
    LOCAL = False
    if LOCAL:
        credential_path = '/Users/battulga/env/gcp/gcp-client.json'
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

    bucket_name = 'battulga-datasets'

    source_blob_name = f'{dataset_name}/dataset.zip'

    destination = 'dataset.zip'

    # download
    storage_client = storage.Client()

    bucket = storage_client.get_bucket(bucket_name)

    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination)

    # extract
    zf = ZipFile(destination)
    zf.extractall()
    zf.close()

    # make generator
    dataset_root = f'{dataset_name}'
    train_dir = f'{dataset_root}/train'
    test_dir = f'{dataset_root}/test'
    val_dir = f'{dataset_root}/val'

    train_size = len(os.listdir(train_dir + '/dog')) + \
        len(os.listdir(train_dir + '/cat'))
    test_size = len(os.listdir(test_dir + '/dog')) + \
        len(os.listdir(test_dir + '/cat'))
    val_size = len(os.listdir(val_dir + '/dog')) + \
        len(os.listdir(val_dir + '/cat'))

    print(f'train size {train_size}')
    print(f'val size {val_size}')
    print(f'test size {test_size}')

    # hyperparams
    epochs = hyperparams.epochs
    batch_size = hyperparams.batch_size
    img_width = hyperparams.img_width
    img_height = hyperparams.img_height
    img_shape = hyperparams.img_shape

    data_gen = ImageDataGenerator(
        rescale=1. / 255, shear_range=0.2, zoom_range=0.2,  horizontal_flip=True)

    mode.fit(train_gen, epochs=epochs)

    test_gen = ImageDataGenerator(rescale=1. / 255)

    val_gen = ImageDataGenerator(rescale=1. / 255)

    def make_gen(path, gen):
        return gen.flow_from_directory(path, target_size=(
            img_width, img_height), batch_size=batch_size, class_mode='binary')

    train_gen = make_gen(train_dir, data_gen)
    test_gen = make_gen(test_dir, test_gen)
    val_gen = make_gen(val_dir, val_gen)

    gen = DatasetGenerator(train_gen=train_gen,
                           test_gen=test_gen,
                           val_gen=val_gen,
                           train_size=train_size,
                           test_size=test_size,
                           val_size=val_size)
    return gen
