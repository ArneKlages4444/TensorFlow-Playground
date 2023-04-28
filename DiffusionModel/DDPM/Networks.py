import tensorflow as tf
import keras
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, Reshape, MaxPool2D, Add, Concatenate, \
    Rescaling, Flatten, Activation
import math


class TimeStepEmbedding(tf.keras.layers.Layer):

    def __init__(self, embedding_dim):
        super(TimeStepEmbedding, self).__init__()
        self.embedding_dim = embedding_dim

    @tf.function
    def call(self, time_steps):
        # from https://arxiv.org/abs/2006.11239
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * - emb)
        emb = tf.cast(time_steps, dtype=tf.float32)[:, None] * emb[None, :]
        emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=1)
        if self.embedding_dim % 2 == 1:  # zero pad
            emb = tf.pad(emb, [[0, 0], [0, 1]])
        return emb


def very_simple_unet(time_steps, learning_rate):
    inp1 = keras.Input(shape=(28, 28, 1))
    inp2 = keras.Input(shape=1)

    emb = TimeStepEmbedding(128)(inp2)
    emb = Flatten()(emb)
    emb = Activation(activation=tf.nn.relu)(emb)

    x1 = Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)(inp1)
    x1 = Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)(x1)

    x_e = Dense(32)(emb)
    x_e = Reshape((1, 1, 32))(x_e)
    x1 = Add()([x1, x_e])

    x2 = MaxPool2D(pool_size=2, strides=2)(x1)
    x2 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu)(x2)
    x2 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu)(x2)

    x_e = Dense(64)(emb)
    x_e = Reshape((1, 1, 64))(x_e)
    x2 = Add()([x2, x_e])

    x3 = MaxPool2D(pool_size=2, strides=2)(x2)
    x3 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu)(x3)
    x3 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu)(x3)

    x_e = Dense(64)(emb)
    x_e = Reshape((1, 1, 64))(x_e)
    x3 = Add()([x3, x_e])

    bottle_neck = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu)(x3)
    bottle_neck = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu)(bottle_neck)
    bottle_neck = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu)(bottle_neck)

    x = Concatenate()([bottle_neck, x3])
    x = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu)(x)

    x_e = Dense(64)(emb)
    x_e = Reshape((1, 1, 64))(x_e)
    x = Add()([x, x_e])

    x = Conv2DTranspose(filters=64, kernel_size=3, padding='same', strides=2, activation=tf.nn.relu)(x)
    x = Concatenate()([x, x2])
    x = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu)(x)

    x_e = Dense(64)(emb)
    x_e = Reshape((1, 1, 64))(x_e)
    x = Add()([x, x_e])

    x = Conv2DTranspose(filters=32, kernel_size=3, padding='same', strides=2, activation=tf.nn.relu)(x)
    x = Concatenate()([x, x1])
    x = Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)(x)

    x_e = Dense(32)(emb)
    x_e = Reshape((1, 1, 32))(x_e)
    x = Add()([x, x_e])

    out = Conv2D(filters=1, kernel_size=1)(x)

    m = keras.Model(inputs=(inp1, inp2), outputs=out)
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    return m
