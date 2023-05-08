import tensorflow as tf
import tensorflow_datasets as tfds


def prepare_mnist_data(ds):
    return ds.map(to_float32).map(set_target).shuffle(10240).batch(256).prefetch(1280)


def to_float32(value, target):
    return tf.cast(value, tf.float32), target


def set_target(value, target):
    return value, value


def get_data_set():
    train_ds, test_ds = tfds.load('MNIST', split=['train', 'test'], as_supervised=True)
    train_dataset = train_ds.apply(prepare_mnist_data)
    test_dataset = test_ds.apply(prepare_mnist_data)
    return train_dataset, test_dataset
