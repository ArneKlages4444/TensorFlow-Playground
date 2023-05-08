import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow import math as tfm
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Conv2DTranspose, Flatten, Reshape, Rescaling
import matplotlib.pyplot as plt


class BetaVAE(tf.keras.Model):

    def __init__(self, latent_size, beta=1, learning_rate=0.001):
        super(BetaVAE, self).__init__()
        self.encoder = EncoderModel(latent_size)
        self.decoder = DecoderModel()
        self.beta = beta
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.compile(optimizer=optimizer)

    @tf.function
    def call(self, inputs):
        mu, sigma = self.encoder(inputs)
        distribution = tfd.Normal(mu, sigma)
        return self.decoder(distribution.sample()), mu, sigma

    @tf.function
    def train_step(self, train_data, reconstruction_loss_fun):
        reconstruction_losses = 0.0
        klds = 0.0
        losses = 0.0
        for data in train_data:
            inp, tar = data
            with tf.GradientTape() as tape:
                img, mu, sigma = self(inp)
                reconstruction_loss = reconstruction_loss_fun(tar, img)
                # KL Divergence of a diagonal multivariate normal and a standard normal distribution
                kld = self.beta * 0.5 * tfm.reduce_sum(
                    (tfm.pow(sigma, 2) + tfm.pow(mu, 2) - 1 - tfm.log(tfm.pow(sigma, 2))))
                loss = 0.5 * (reconstruction_loss + kld)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            reconstruction_losses += reconstruction_loss
            klds += kld
            losses += loss
        return losses, reconstruction_losses, klds

    def train(self, train_dataset, test_dataset, epochs=100):
        self.show_test_img(test_dataset)
        tf.keras.backend.clear_session()
        loss_fun = tf.keras.losses.MeanSquaredError()
        for epoch in range(epochs):
            losses, reconstruction_losses, klds = self.train_step(train_dataset, loss_fun)
            batches = float(train_dataset.__len__())
            print(
                f"epoch {epoch}, loss={losses / batches}, reconstruction_loss={reconstruction_losses / batches}, KLD={klds / batches}")
            self.show_test_img(test_dataset)

    def show_test_img(self, data_set):
        images = None
        for d in data_set.take(1):
            images, _ = d
        plt.imshow(images[0].numpy().astype("uint8")[:, :, 0], cmap='gray')
        plt.show()
        x, _, _ = self(images)
        plt.imshow(x[0].numpy().astype("uint8")[:, :, 0], cmap='gray')
        plt.show()


class EncoderModel(tf.keras.Model):

    def __init__(self, latent_size):
        super(EncoderModel, self).__init__()
        self.scaling = Rescaling(1. / 255)
        self.conv_layer1 = Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.pooling1 = MaxPool2D(pool_size=2, strides=2)
        self.conv_layer2 = Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.pooling2 = MaxPool2D(pool_size=2, strides=2)
        self.conv_layer3 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.flatten = Flatten()
        self.mu = Dense(latent_size, activation=None)
        self.sigma = Dense(latent_size, activation=tf.nn.softplus)

    @tf.function
    def call(self, inputs):
        x = self.scaling(inputs)
        x = self.conv_layer1(x)
        x = self.pooling1(x)
        x = self.conv_layer2(x)
        x = self.pooling2(x)
        x = self.conv_layer3(x)
        x = self.flatten(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        return mu, sigma


class DecoderModel(tf.keras.Model):

    def __init__(self):
        super(DecoderModel, self).__init__()
        self.dense = Dense(3136, activation=tf.nn.relu)  # 7 * 7 * 64 = 3.136
        self.reshape = Reshape((7, 7, 64))
        self.conv_t1 = Conv2DTranspose(filters=64, kernel_size=3, padding='same', strides=2, activation=tf.nn.relu)
        self.conv_t2 = Conv2DTranspose(filters=32, kernel_size=3, padding='same', strides=2, activation=tf.nn.relu)
        self.conv = Conv2D(filters=1, kernel_size=3, padding='same', activation=tf.nn.sigmoid)
        self.out = Rescaling(255)

    @tf.function
    def call(self, inputs):
        x = self.dense(inputs)
        x = self.reshape(x)
        x = self.conv_t1(x)
        x = self.conv_t2(x)
        x = self.conv(x)
        x = self.out(x)
        return x
