import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Conv2D, Conv2DTranspose, Flatten, Reshape, BatchNormalization, \
    Rescaling, Dropout
import matplotlib.pyplot as plt


class GAN:

    def __init__(self, generator_learning_rate=0.0001, discriminator_learning_rate=0.0001):
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.generator.compile(optimizer=tf.keras.optimizers.Adam(generator_learning_rate))
        self.discriminator.compile(optimizer=tf.keras.optimizers.Adam(discriminator_learning_rate))

    @tf.function
    def train_step(self, data_set):
        generator_losses = 0.0
        discriminator_losses = 0.0
        for data in data_set:
            with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
                batch_size = tf.shape(data)[0]
                noise = self.generator(tf.random.normal([batch_size, self.generator.input_size]))
                fake_data_pred = self.discriminator(noise, True)
                real_data_pred = self.discriminator(data, True)
                loss_fun = tf.keras.losses.BinaryCrossentropy()
                generator_loss = loss_fun(tf.ones_like(fake_data_pred), fake_data_pred)
                discriminator_loss = (loss_fun(tf.ones_like(real_data_pred), real_data_pred) + loss_fun(
                    tf.zeros_like(fake_data_pred), fake_data_pred))
            discriminator_gradients = discriminator_tape.gradient(discriminator_loss,
                                                                  self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(discriminator_gradients, self.discriminator.trainable_variables))
            generator_gradients = generator_tape.gradient(generator_loss, self.generator.trainable_variables)
            self.generator.optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
            generator_losses += generator_loss
            discriminator_losses += discriminator_loss
        return generator_losses, discriminator_losses

    def train(self, data_set, num_epochs=100):
        self.show_example_img()
        for epoch in range(num_epochs):
            generator_loss, discriminator_loss = self.train_step(data_set)
            batches = float(data_set.__len__())
            print(
                f"epoch {epoch}, generator_loss={generator_loss / batches}, discriminator_loss={discriminator_loss / batches}")
            self.show_example_img()

    def show_example_img(self):
        b = self.generator(tf.random.normal([1, self.generator.input_size]))
        plt.imshow(b[0].numpy().astype("uint8")[:, :, 0], cmap='gray')
        plt.show()


class Discriminator(tf.keras.Model):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.scaling = Rescaling(1. / 127.5, offset=-1)
        self.conv_layer1 = Conv2D(filters=32, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.conv_layer1 = Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        self.dropout1 = Dropout(0.2)
        self.conv_layer2 = Conv2D(filters=64, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.conv_layer2 = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        self.dropout2 = Dropout(0.2)
        self.conv_layer3 = Conv2D(filters=128, kernel_size=3, padding='same', activation=tf.nn.relu)
        self.conv_layer3 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)
        self.dropout3 = Dropout(0.2)
        self.flatten = Flatten()
        self.out = Dense(1, activation=tf.nn.sigmoid)  # activation=None)

    @tf.function
    def call(self, inputs, is_training=False):
        x = self.scaling(inputs)
        x = self.conv_layer1(x)
        x = self.dropout1(x, training=is_training)
        x = self.conv_layer2(x)
        x = self.dropout2(x, training=is_training)
        x = self.conv_layer3(x)
        x = self.dropout3(x, training=is_training)
        x = self.flatten(x)
        return self.out(x)


class Generator(tf.keras.Model):

    def __init__(self):
        super(Generator, self).__init__()
        self.input_size = 100
        self.dense = Dense(7 * 7 * 256)
        self.reshape = Reshape((7, 7, 256))
        self.b_norm1 = BatchNormalization()
        self.activation1 = Activation(activation=tf.nn.relu)
        self.conv1 = Conv2D(filters=256, kernel_size=3, padding='same')
        self.b_norm2 = BatchNormalization()
        self.activation2 = Activation(activation=tf.nn.relu)
        self.conv_t1 = Conv2DTranspose(filters=128, kernel_size=4, padding='same', strides=2)
        self.conv2 = Conv2D(filters=128, kernel_size=3, padding='same')
        self.b_norm3 = BatchNormalization()
        self.activation3 = Activation(activation=tf.nn.relu)
        self.conv_t2 = Conv2DTranspose(filters=64, kernel_size=4, padding='same', strides=2)
        self.conv3 = Conv2D(filters=64, kernel_size=3, padding='same')
        self.b_norm4 = BatchNormalization()
        self.activation4 = Activation(activation=tf.nn.relu)
        self.conv = Conv2D(filters=1, kernel_size=3, padding='same', activation=tf.nn.sigmoid)  # activation=tf.nn.tanh
        self.out = Rescaling(255)

    @tf.function
    def call(self, inputs):
        x = self.dense(inputs)
        x = self.reshape(x)
        x = self.b_norm1(x)
        x = self.activation1(x)
        x = self.conv1(x)
        x = self.b_norm2(x)
        x = self.activation2(x)
        x = self.conv_t1(x)
        x = self.conv2(x)
        x = self.b_norm3(x)
        x = self.activation3(x)
        x = self.conv_t2(x)
        x = self.conv3(x)
        x = self.b_norm4(x)
        x = self.activation4(x)
        x = self.conv(x)
        x = self.out(x)
        return x
