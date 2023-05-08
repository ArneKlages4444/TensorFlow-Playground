import tensorflow as tf
import os
import numpy as np
import urllib


class CandleDataSet:

    def __init__(self, batch_size=256):
        self.batch_size = batch_size

    def prepare_data(self, ds):
        return ds.map(self.to_float32).map(self.reshape).cache().shuffle(1000).batch(self.batch_size).prefetch(32)

    def to_float32(self, value):
        return tf.cast(value, tf.float32)

    def reshape(self, value):
        return tf.reshape(value, (28, 28, 1))

    def get_data(self):
        category = 'candle'
        if not os.path.isdir('npy_files'):
            os.mkdir('npy_files')
        url = f'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{category}.npy'
        urllib.request.urlretrieve(url, f'npy_files/{category}.npy')
        images = np.load(f'npy_files/{category}.npy')
        train_ds = tf.data.Dataset.from_tensor_slices(images)
        return train_ds.apply(self.prepare_data)
